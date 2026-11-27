//! The states the PC provider records with and the states the title gets back.

#[derive(Clone, Copy)]
pub(super) struct Binding {
    /// Host resource identity and its last state on this list.
    pub(super) before: (usize, Option<u32>),
    /// The state to restore after a successful dispatch.
    pub(super) declared: u32,
    pub(super) used: u32,
    /// An observed or declared state the caller can restore after a failure.
    pub(super) on_failure: Option<u32>,
}

pub(super) enum SetupError {
    Aliased(usize),
    Input(usize),
}

/// Preparing an aliased input can move another binding into a copy state.
/// Bind the provider's states after every preparation, and restore the title's
/// states even if only some of the preparations or bindings have run.
pub(super) fn with_resource_states<T, E: From<SetupError>>(
    bindings: [Option<Binding>; 7],
    mut prepare: impl FnMut(usize) -> bool,
    mut bind: impl FnMut(usize, u32) -> Result<T, E>,
    mut current: impl FnMut(usize) -> Option<(usize, Option<u32>)>,
    mut restore: impl FnMut(usize, u32),
    record: impl FnOnce(&[Option<T>; 7]) -> Result<(), E>,
) -> Result<(), E> {
    const OUTPUT: usize = 6;
    for (index, binding) in bindings.iter().enumerate() {
        let Some(b) = binding else { continue };
        if bindings[..index].iter().flatten().any(|other| {
            other.before.0 == b.before.0
                && (index == OUTPUT
                    || other.declared != b.declared
                    || other.used != b.used
                    || other.on_failure.zip(b.on_failure).is_some_and(|(a, b)| a != b))
        }) {
            return Err(SetupError::Aliased(index).into());
        }
    }
    let mut all_bound = false;
    let outcome = (|| {
        if let Some(index) = (0..OUTPUT).find(|&i| bindings[i].is_some() && !prepare(i)) {
            return Err(SetupError::Input(index).into());
        }
        let mut textures = std::array::from_fn(|_| None);
        for (index, binding) in bindings.iter().enumerate() {
            if let Some(b) = binding {
                textures[index] = Some(bind(index, b.used)?);
            }
        }
        all_bound = true;
        record(&textures)
    })();
    for (index, binding) in bindings.iter().enumerate() {
        let Some(b) = binding else { continue };
        let after = if outcome.is_ok() { Some(b.declared) } else { b.on_failure };
        let Some(after) = after else { continue };
        // Leave untouched bindings alone even when their declarations differ.
        let changed = if all_bound {
            after != b.used
        } else {
            current(index)
                .is_some_and(|now| now != b.before && now.1.is_some_and(|state| state != after))
        };
        if changed {
            restore(index, after);
        }
    }
    outcome
}
