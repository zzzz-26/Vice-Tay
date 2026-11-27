# Repository workflow

Keep the primary local checkout and the GitHub default branch in sync in both directions. Pull changes merged on GitHub into the primary checkout. Bring versioned local changes back through a pull request before merging them into the default branch. If work starts in another checkout, bring the completed changes into the primary checkout.

Before closing out a change, run `git fetch` and `git status`, compare the local and remote default branches, and verify that they agree after the merge. Preserve work in progress. Document any remaining divergence and what is needed to resolve it.

## Visual quality review

Record every visual verification from start to finish and review the entire video before reporting results. Keep the recording with the test evidence and note timestamps for visible changes and artifacts.

For comparisons, use the same scene, camera, resolution, upscaler and frame generation settings. Check colour, lighting, textures, reflections, the HUD and temporal artifacts alongside rendered frame rate and frame times. State any trade-off between image quality and performance.
