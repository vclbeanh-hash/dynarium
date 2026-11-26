// SPDX-License-Identifier: GPL-2.0-or-later
#include "native_kernel_layout.hpp"

#include "host_environment.hpp"
#include "memory/mmu/address_layout.hpp"
#include "memory/shared_backing.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>

#include <sys/mman.h>
#include <unistd.h>

namespace uml::runtime {
namespace {

struct ProcessWindow final {
	std::mutex mutex;
	std::uintptr_t begin{};
	std::uintptr_t end{};
	bool fresh{};
};

/* Never destroyed: a guest thread can still run when the process ends. */
[[clang::no_destroy]] constinit ProcessWindow process_window;

[[nodiscard]] NativeKernelLayout reserved_layout(std::uintptr_t base,
	std::size_t length) noexcept
{
	return {
		.load_bias = base - DARWIN_UML_NATIVE_LINK_ADDRESS,
		.window_start = base,
		.window_end = base + length,
		.valid = true,
		.reserved = true,
	};
}

/* A window is a multiple of the guest page, the host maps whole host pages,
 * and the reservation is what the host mapped. */
[[nodiscard]] std::size_t host_reservation_length(std::size_t length) noexcept
{
	const auto page = static_cast<std::size_t>(getpagesize());
	return (length + page - 1U) & ~(page - 1U);
}

/* Reserves [end, new_end) where the process's window ends, if the host has it
 * free; nothing is changed otherwise. */
[[nodiscard]] bool extend_reservation(std::uintptr_t end,
	std::uintptr_t new_end) noexcept
{
	const std::size_t length = new_end - end;
	void *const tail = mmap(reinterpret_cast<void *>(end), length, PROT_NONE,
		MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
	if (tail == MAP_FAILED)
		return false;
	if (reinterpret_cast<std::uintptr_t>(tail) != end) {
		munmap(tail, length);
		return false;
	}
	return true;
}

} // namespace

NativeKernelPlacement native_kernel_placement(
	std::size_t guest_memory) noexcept
{
#if DARWIN_UML_NATIVE_HIGH_PLACEMENT
	return {MacosNativeKernelGeometry::derive_at(
			guest_memory, DARWIN_UML_MACOS_NATIVE_ADDRESS),
		WindowPlacement::exact};
#else
	/* The same iOS build runs on an Apple silicon Mac in Designed for iPad
	 * mode, where the process is a macOS process: the device window below
	 * the signed iOS envelope is already occupied there, and Darwin answers
	 * the loader's address hint with somewhere else entirely, which the
	 * loader must reject. Placement follows the address space the process
	 * actually has rather than the SDK it was built against.
	 */
	if (::uml::platform::running_ios_app_on_macos()) {
		return {MacosNativeKernelGeometry::derive_at(
				guest_memory, DARWIN_UML_MACOS_NATIVE_ADDRESS),
			WindowPlacement::exact};
	}
	return {DeviceNativeKernelGeometry::derive(guest_memory),
		WindowPlacement::anywhere};
#endif
}

NativeKernelLayout prepare_native_kernel_layout(
	std::size_t guest_memory) noexcept
{
	const auto product = native_kernel_placement(guest_memory);
	if (!product.geometry.valid)
		return product.geometry;
	const auto layout = reserve_process_native_window(
		product.geometry, product.placement);
	if (!layout.valid) {
		std::fprintf(stderr,
			"UML: cannot reserve a %llu-byte native kernel window above "
			"%#llx (check com.apple.developer.kernel.extended-virtual-"
			"addressing entitlement)\n",
			static_cast<unsigned long long>(
				product.geometry.window_end -
				product.geometry.window_start),
			static_cast<unsigned long long>(
				DARWIN_UML_NATIVE_LINK_ADDRESS));
		return {.reservation_failed = true};
	}
	return layout;
}

NativeKernelLayout reserve_process_native_window(
	const NativeKernelLayout &geometry, WindowPlacement placement) noexcept
{
	if (!geometry.valid || geometry.window_end <= geometry.window_start)
		return {};
	const std::size_t length = static_cast<std::size_t>(
		geometry.window_end - geometry.window_start);
	std::lock_guard lock{process_window.mutex};
	const bool held = process_window.end != 0;
	if (!held && placement == WindowPlacement::anywhere) {
		/* Reserve a generous PROT_NONE | MAP_NORESERVE VA window upfront on
		 * iOS so later VM restarts at higher RAM reuse the same base without
		 * unmapping permanent signed kernel text aliases. */
		constexpr std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
		constexpr std::array<std::size_t, 3> upfront_guests {
			4ULL * gib, 3ULL * gib, 2ULL * gib};
		for (const auto guest_ceiling : upfront_guests) {
			const auto candidate =
				DeviceNativeKernelGeometry::derive(guest_ceiling);
			if (!candidate.valid ||
			    candidate.window_end <= candidate.window_start)
				continue;
			const std::size_t candidate_length = static_cast<std::size_t>(
				candidate.window_end - candidate.window_start);
			if (candidate_length < length)
				continue;
			const auto window =
				reserve_native_kernel_window(candidate, placement);
			if (!window.valid)
				continue;
			process_window.begin = window.window_start;
			process_window.end = window.window_start +
				host_reservation_length(candidate_length);
			process_window.fresh = true;
			return reserved_layout(process_window.begin, length);
		}
	}
	/* A window the host places is the window wherever it is; one at an exact
	 * address must be at that address. */
	const bool fits_place = held &&
		(placement == WindowPlacement::anywhere ||
		 process_window.begin == geometry.window_start);
	if (fits_place &&
	    process_window.end - process_window.begin >= length)
		return reserved_layout(process_window.begin, length);
	if (fits_place) {
		const std::size_t aligned_length = host_reservation_length(length);
		if (aligned_length <=
		    static_cast<std::size_t>(
			    DARWIN_UML_GUEST_ADDRESS_LIMIT - process_window.begin)) {
			const std::uintptr_t new_end =
				process_window.begin + aligned_length;
			if (extend_reservation(process_window.end, new_end)) {
				process_window.end = new_end;
				return reserved_layout(process_window.begin, length);
			}
		}
		if (placement == WindowPlacement::exact)
			return {};
		if (::uml::memory::has_kept_host_ranges()) {
			std::fprintf(stderr,
				"UML: cannot grow native kernel window after signed "
				"kernel text was locked in place (held %llu MiB, "
				"wanted %llu MiB); reduce RAM or restart the app\n",
				static_cast<unsigned long long>(
					(process_window.end - process_window.begin) >> 20),
				static_cast<unsigned long long>(length >> 20));
			return {};
		}
	}
	const auto window = reserve_native_kernel_window(geometry, placement);
	if (!window.valid)
		return {};
	/* The new reservation is held first, so a failure above leaves the old
	 * one. The old holds nothing: it was a window for a guest the process no
	 * longer has, handed back or never booted, and no permanent signed text
	 * ranges remain inside it. */
	if (held)
		munmap(reinterpret_cast<void *>(process_window.begin),
			process_window.end - process_window.begin);
	process_window.begin = window.window_start;
	process_window.end = window.window_start + host_reservation_length(length);
	process_window.fresh = true;
	return window;
}

ProcessNativeWindow process_native_window() noexcept
{
	std::lock_guard lock{process_window.mutex};
	return {process_window.begin, process_window.end};
}

bool claim_fresh_native_kernel_window(
	std::uintptr_t address, std::size_t length) noexcept
{
	std::lock_guard lock{process_window.mutex};
	if (!process_window.fresh || length == 0 ||
	    address < process_window.begin || address >= process_window.end ||
	    length > process_window.end - address)
		return false;
	/* The reservation was allocated as one intact PROT_NONE mapping, and
	 * nobody has installed the guest kernel into it yet. Consume once:
	 * later launches must explicitly reset their old kernel data pages. */
	process_window.fresh = false;
	return true;
}

NativeKernelLayout reserve_native_kernel_window(
	const NativeKernelLayout &geometry, WindowPlacement placement) noexcept
{
	if (!geometry.valid || geometry.window_end <= geometry.window_start)
		return {};
	const std::size_t length = static_cast<std::size_t>(
		geometry.window_end - geometry.window_start);
	/* Reserve the window before the guest is brought up so nothing else can
	 * take part of it. The address the geometry names is tried first, so a
	 * host that still offers it keeps the established layout. On iOS 16–25,
	 * top-packed windows near 62 GiB may collide with XNU VM reservations, so
	 * downward probes toward the 48 GiB link floor and high-VA hints find an
	 * open slot when the packed hint is moved low. iOS 27 places unhinted
	 * mappings above 48 GiB automatically. A macOS kernel is built for its
	 * address and gets no other.
	 *
	 * Nothing inside the window is released while it lives, not even for
	 * the kernel image: Darwin prefers a freshly freed range for its next
	 * unhinted allocation, and a host allocation that landed there would be
	 * replaced by the next MAP_FIXED guest mapping without either side being
	 * told. The loader and guest RAM map over the reservation in place.
	 */
	const std::size_t reservation = host_reservation_length(length);
	if (reservation == 0 || reservation > DARWIN_UML_GUEST_ADDRESS_LIMIT)
		return {};

	const auto valid_base = [&](std::uintptr_t base) noexcept {
		return (base & (DARWIN_UML_GUEST_PAGE_SIZE - 1U)) == 0U &&
			base >= DARWIN_UML_NATIVE_LINK_ADDRESS &&
			reservation <= DARWIN_UML_GUEST_ADDRESS_LIMIT - base;
	};
	const auto try_hint = [&](std::uintptr_t hint) noexcept
		-> NativeKernelLayout {
		if (hint != 0U &&
		    (hint < DARWIN_UML_NATIVE_LINK_ADDRESS ||
		     hint > DARWIN_UML_GUEST_ADDRESS_LIMIT - reservation))
			return {};
		void *const mapped = mmap(reinterpret_cast<void *>(hint),
			reservation, PROT_NONE,
			MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
		if (mapped == MAP_FAILED)
			return {};
		const auto base = reinterpret_cast<std::uintptr_t>(mapped);
		if (placement == WindowPlacement::exact) {
			if (base != geometry.window_start || !valid_base(base)) {
				munmap(mapped, reservation);
				return {};
			}
			return reserved_layout(base, length);
		}
		if (!valid_base(base)) {
			munmap(mapped, reservation);
			return {};
		}
		return reserved_layout(base, length);
	};

	if (placement == WindowPlacement::exact)
		return try_hint(geometry.window_start);

	const auto host_page = static_cast<std::uintptr_t>(getpagesize());
	const auto align_down_host = [host_page](std::uintptr_t address) noexcept {
		return address & ~(host_page - 1U);
	};
	const std::uintptr_t initial_hint = std::max<std::uintptr_t>(
		DARWIN_UML_NATIVE_LINK_ADDRESS,
		align_down_host(geometry.window_start));

	if (const auto found = try_hint(initial_hint); found.valid)
		return found;

	constexpr std::uintptr_t probe_step = 64ULL * 1024ULL * 1024ULL;
	if (initial_hint > DARWIN_UML_NATIVE_LINK_ADDRESS) {
		std::uintptr_t hint = initial_hint;
		for (;;) {
			hint = hint - DARWIN_UML_NATIVE_LINK_ADDRESS > probe_step
				? align_down_host(hint - probe_step)
				: static_cast<std::uintptr_t>(
					  DARWIN_UML_NATIVE_LINK_ADDRESS);
			if (const auto found = try_hint(hint); found.valid)
				return found;
			if (hint == static_cast<std::uintptr_t>(
					    DARWIN_UML_NATIVE_LINK_ADDRESS))
				break;
		}
	}

	constexpr std::uintptr_t gib = 1024ULL * 1024ULL * 1024ULL;
	constexpr std::array<std::uintptr_t, 6> high_hints {
		64ULL * gib,
		80ULL * gib,
		96ULL * gib,
		128ULL * gib,
		256ULL * gib,
		512ULL * gib,
	};
	for (const auto hint : high_hints) {
		if (const auto found = try_hint(hint); found.valid)
			return found;
	}

	return try_hint(0U);
}

NativeKernelLayoutScope::NativeKernelLayoutScope(
	const NativeKernelLayout &layout) noexcept
{
	if (!layout.valid)
		return;
	entered_ = ::uml::memory::configure_native_address_window(
		layout.window_start, layout.window_end);
}

NativeKernelLayoutScope::~NativeKernelLayoutScope()
{
	if (entered_)
		::uml::memory::reset_native_address_window();
}

} // namespace uml::runtime
