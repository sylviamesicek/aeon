use bumpalo::Bump;
use thread_local::ThreadLocal;

pub struct MeshStore {
    tl: ThreadLocal<Bump>,
}
