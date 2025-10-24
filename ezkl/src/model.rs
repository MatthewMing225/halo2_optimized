use halo2_proofs::{arithmetic::Field, circuit::Region};

use crate::circuit::ops::region::RegionCtx;

/// Placeholder model structure that demonstrates how `RegionCtx` is consumed
/// by layout helpers.
#[derive(Default)]
pub struct Model;

impl Model {
    /// Synthesises assignments within the provided region.
    pub fn layout<'a, F: Field>(&self, region: Region<'a, F>) {
        let ctx = RegionCtx::new(region);
        let _ = ctx.with_region(|_| {
            // In the real implementation this would assign advice/fixed cells.
        });
    }

    /// Synthesises a dummy layout that operates without a backing region.
    pub fn dummy_layout<'a, F: Field>(&self) {
        let mut ctx: RegionCtx<'a, F> = RegionCtx::new_dummy();
        ctx.advance(1);
        let _ = ctx.offset();
    }
}
