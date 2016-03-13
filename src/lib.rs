#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(dropck_parametricity)]
#![feature(filling_drop)]
#![feature(heap_api)]
#![feature(oom)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag)]

extern crate alloc;

mod hash {
    mod table;
}
