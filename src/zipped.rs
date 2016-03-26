// Copyright © 2016  Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use std::mem::{align_of, size_of};
use std::ptr;


pub trait ZippedPtrs : Sized {

    type Values;

    unsafe fn offset(&self, count: isize) -> Self;

    unsafe fn write(&self, data: Self::Values);

    unsafe fn read(&self) -> Self::Values;

    unsafe fn replace(&self, newval: Self::Values) -> Self::Values;

    unsafe fn copy_nonoverlapping(src: &Self, dst: &Self, count: usize);

    unsafe fn from_unzipped_buf(buffer: *mut u8,
                                offset: usize,
                                length: usize)
                                -> Self;

    fn alloc_size_append(offset: usize, length: usize) -> Option<usize>;
    fn alloc_size_append_unchecked(offset: usize, length: usize) -> usize;
    fn align() -> usize;
}

pub trait AsZippedRefs<'a> : ZippedPtrs {

    type Refs: 'a;
    type MutRefs: 'a;

    unsafe fn as_refs(&self) -> Self::Refs;
    unsafe fn as_mut_refs(&self) -> Self::MutRefs;
}

pub trait CloneZipped : ZippedPtrs {
    unsafe fn clone_zipped(&self) -> Self::Values;
}

pub struct LonePtr<T> {
    ptr: *mut T
}

pub struct PairPtrs<K, V> {
    key: *mut K,
    val: *mut V
}

// The zipped pointer types are copyable akin to raw pointers:
// as long as the access interface is unsafe, the instances can be safely
// copied.

impl<T> Copy for LonePtr<T> {}
impl<T> Clone for LonePtr<T> {
    fn clone(&self) -> LonePtr<T> { *self }
}

impl<K, V> Copy for PairPtrs<K, V> {}
impl<K, V> Clone for PairPtrs<K, V> {
    fn clone(&self) -> PairPtrs<K, V> { *self }
}

/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// Note that overflows are not checked by this function; use
/// `checked_round_up_to_next` when it's not certain that the result can be
/// represented by `usize`.  
///
/// # Panics
///
/// Panics if `target_alignment` is not a power of two.
#[inline]
fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}

/// Rounds up to a multiple of a power of two. Returns the closest multiple of
/// `target_alignment` that is higher or equal to  `unrounded`, if such a
/// value can be represented by `usize`. Otherwise, returns `None`.
///
/// # Panics
///
/// Panics if `target_alignment` is not a power of two.
#[inline]
fn checked_round_up_to_next(unrounded: usize, target_alignment: usize)
                            -> Option<usize> {
    assert!(target_alignment.is_power_of_two());

    // target_alignment is positive,
    // so (target_alignment - 1) does not need overflow checking

    unrounded.checked_add(target_alignment - 1)
        .map(|acc| {
            acc & !(target_alignment - 1)
        })
}

#[test]
fn test_rounding() {
    use std::usize;

    assert_eq!(round_up_to_next(0, 4), 0);
    assert_eq!(checked_round_up_to_next(0, 4), Some(0));
    assert_eq!(round_up_to_next(1, 4), 4);
    assert_eq!(checked_round_up_to_next(1, 4), Some(4));
    assert_eq!(round_up_to_next(2, 4), 4);
    assert_eq!(checked_round_up_to_next(2, 4), Some(4));
    assert_eq!(round_up_to_next(3, 4), 4);
    assert_eq!(checked_round_up_to_next(3, 4), Some(4));
    assert_eq!(round_up_to_next(4, 4), 4);
    assert_eq!(checked_round_up_to_next(4, 4), Some(4));
    assert_eq!(round_up_to_next(5, 4), 8);
    assert_eq!(checked_round_up_to_next(5, 4), Some(8));
    assert_eq!(checked_round_up_to_next(usize::MAX & !3, 4),
               Some(usize::MAX & !3));
    assert_eq!(checked_round_up_to_next((usize::MAX & !3) + 1, 4), None);
    assert_eq!(checked_round_up_to_next(usize::MAX, 4), None);
    assert_eq!(checked_round_up_to_next(usize::MAX, 2), None);
}

#[inline]
fn unzipped_size_append_unchecked(offset: usize, size: usize, align: usize)
                                  -> usize {
    let splice_offset = round_up_to_next(offset, align);
    let total_size = splice_offset + size;

    total_size
}

#[inline]
fn unzipped_size_append_checked(offset: usize, size: usize, align: usize)
                                -> Option<usize> {
    checked_round_up_to_next(offset, align)
        .and_then(|aligned_offset| {
            aligned_offset.checked_add(size)
        })
}

impl<T> ZippedPtrs for LonePtr<T> {

    type Values  = T;

    unsafe fn offset(&self, count: isize) -> LonePtr<T> {
        LonePtr {
            ptr: self.ptr.offset(count)
        }
    }

    unsafe fn write(&self, values: T) {
        ptr::write(self.ptr, values)
    }

    unsafe fn read(&self) -> T {
        ptr::read(self.ptr)
    }

    unsafe fn replace(&self, values: T) -> T {
        ptr::replace(self.ptr, values)
    }

    unsafe fn copy_nonoverlapping(src: &Self, dst: &Self, count: usize) {
        ptr::copy_nonoverlapping(src.ptr, dst.ptr, count);
    }

    unsafe fn from_unzipped_buf(buffer: *mut u8,
                                offset: usize,
                                _length: usize)
                                -> LonePtr<T> {
        let vals_offset = round_up_to_next(offset, align_of::<T>());
        LonePtr { ptr: buffer.offset(vals_offset as isize) as *mut T }
    }

    fn alloc_size_append_unchecked(offset: usize, length: usize) -> usize {
        let size = length * size_of::<T>();
        unzipped_size_append_unchecked(offset, size, align_of::<T>())
    }

    fn alloc_size_append(offset: usize, length: usize) -> Option<usize> {
        length.checked_mul(size_of::<T>())
            .and_then(|size| {
                unzipped_size_append_checked(offset, size, align_of::<T>())
            })
    }

    fn align() -> usize { align_of::<T>() }
}

impl<K, V> ZippedPtrs for PairPtrs<K, V> {

    type Values = (K, V);

    unsafe fn offset(&self, count: isize) -> PairPtrs<K, V> {
        PairPtrs {
            key: self.key.offset(count),
            val: self.val.offset(count)
        }
    }

    unsafe fn write(&self, entry: (K, V)) {
        ptr::write(self.key, entry.0);
        ptr::write(self.val, entry.1);
    }

    unsafe fn read(&self) -> (K, V) {
        (ptr::read(self.key), ptr::read(self.val))
    }

    unsafe fn replace(&self, entry: (K, V)) -> (K, V) {
        let old_key = ptr::replace(self.key, entry.0);
        let old_val = ptr::replace(self.val, entry.1);

        (old_key, old_val)
    }

    unsafe fn copy_nonoverlapping(src: &Self, dst: &Self, count: usize) {
        ptr::copy_nonoverlapping(src.key, dst.key, count);
        ptr::copy_nonoverlapping(src.val, dst.val, count);
    }

    unsafe fn from_unzipped_buf(buffer: *mut u8,
                                offset: usize,
                                length: usize)
                                -> PairPtrs<K, V> {
        let keys_offset = round_up_to_next(offset, align_of::<K>());
        let keys_size = length * size_of::<K>();
        let end_of_keys = keys_offset + keys_size;
        let vals_offset = round_up_to_next(end_of_keys, align_of::<V>());
        PairPtrs {
            key: buffer.offset(keys_offset as isize) as *mut K,
            val: buffer.offset(vals_offset as isize) as *mut V
        }
    }

    fn alloc_size_append_unchecked(offset: usize, length: usize) -> usize {
        let keys_size = length * size_of::<K>();
        let vals_size = length * size_of::<V>();
        let end_of_keys =
            unzipped_size_append_unchecked(offset, keys_size,
                                           align_of::<K>());
        let end_of_vals =
            unzipped_size_append_unchecked(end_of_keys, vals_size,
                                           align_of::<V>());

        end_of_vals
    }

    fn alloc_size_append(offset: usize, length: usize) -> Option<usize> {
        let sizes_checked = (length.checked_mul(size_of::<K>()),
                             length.checked_mul(size_of::<V>()));
        if let (Some(keys_size), Some(vals_size)) = sizes_checked {
            unzipped_size_append_checked(offset, keys_size,
                                         align_of::<K>())
                .and_then(|end_of_keys| {
                    unzipped_size_append_checked(end_of_keys, vals_size,
                                                 align_of::<V>())
                })
        } else {
            None
        }
    }

    fn align() -> usize {
        cmp::max(align_of::<K>(), align_of::<V>())
    }
}

impl<'a, T: 'a> AsZippedRefs<'a> for LonePtr<T> {

    type Refs    = &'a T;
    type MutRefs = &'a mut T;

    unsafe fn as_refs(&self) -> &'a T {
        &*self.ptr
    }

    unsafe fn as_mut_refs(&self) -> &'a mut T {
        &mut *self.ptr
    }
}

impl<'a, K: 'a, V: 'a> AsZippedRefs<'a> for PairPtrs<K, V> {

    type Refs    = (&'a K, &'a V);
    type MutRefs = (&'a mut K, &'a mut V);

    unsafe fn as_refs(&self) -> (&'a K, &'a V) {
        (&*self.key, &*self.val)
    }

    unsafe fn as_mut_refs(&self) -> (&'a mut K, &'a mut V) {
        (&mut *self.key, &mut *self.val)
    }
}

impl <T: Clone> CloneZipped for LonePtr<T> {
    unsafe fn clone_zipped(&self) -> T {
        (*self.ptr).clone()
    }
}

impl <K: Clone, V: Clone> CloneZipped for PairPtrs<K, V> {
    unsafe fn clone_zipped(&self) -> (K, V) {
        ((*self.key).clone(), (*self.val).clone())
    }
}
