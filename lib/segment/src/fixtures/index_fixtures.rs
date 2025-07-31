use std::borrow::Cow;

use bitvec::prelude::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use rand::Rng;

use crate::data_types::named_vectors::CowVector;
use crate::data_types::vectors::{DenseVector, VectorElementType, VectorRef};
use crate::index::hnsw_index::point_scorer::FilteredScorer;
use crate::types::Distance;
use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
use crate::vector_storage::{VectorStorage, VectorStorageEnum};

pub fn random_vector<R: Rng + ?Sized>(rnd_gen: &mut R, size: usize) -> DenseVector {
    (0..size).map(|_| rnd_gen.random_range(-1.0..1.0)).collect()
}

pub struct TestRawScorerProducer {
    pub vector_storage: VectorStorageEnum,
    pub deleted_points: BitVec,
    pub distance: Distance,
}

impl TestRawScorerProducer {
    pub fn new<R>(dim: usize, num_vectors: usize, distance: Distance, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let mut vector_storage = new_volatile_dense_vector_storage(dim, distance);
        let hw_counter = HardwareCounterCell::new();
        for offset in 0..num_vectors {
            let rnd_vec = random_vector(rng, dim);
            let rnd_vec = distance.preprocess_vector::<VectorElementType>(rnd_vec);
            vector_storage
                .insert_vector(
                    offset as PointOffsetType,
                    VectorRef::from(&rnd_vec),
                    &hw_counter,
                )
                .unwrap();
        }

        TestRawScorerProducer {
            vector_storage,
            deleted_points: BitVec::repeat(false, num_vectors),
            distance,
        }
    }

    pub fn get_vector(&self, key: PointOffsetType) -> Cow<[VectorElementType]> {
        match self.vector_storage.get_vector(key) {
            CowVector::Dense(cow) => cow,
            _ => unreachable!("Expected vector storage to be dense"),
        }
    }

    pub fn get_scorer(&self, query: DenseVector) -> FilteredScorer<'_> {
        let query = self
            .distance
            .preprocess_vector::<VectorElementType>(query)
            .into();
        FilteredScorer::new_for_test(query, &self.vector_storage, &self.deleted_points)
    }
}
