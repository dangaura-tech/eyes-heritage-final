## Ensemble variants
| variant | M_in | M_out | S | S 95% CI (cell bootstrap) | F1 constr/destr/soil/no_ch (LOSO) | Aleppo | S organizer pts, original labels |
|---|---|---|---|---|---|---|---|
| a0_equal_argmax | 0.7300 | 0.6408 | **0.6825** | 0.643–0.717 | [0.74, 0.503, 0.512, 0.807] | 0.761 | 0.6836 |
| a_equal_nested_tb | 0.7172 | 0.6278 | **0.6696** | 0.629–0.706 | [0.731, 0.473, 0.506, 0.801] | 0.771 | 0.6711 |
| b_fully_nested | 0.7042 | 0.6159 | **0.6571** | 0.616–0.692 | [0.725, 0.462, 0.472, 0.805] | 0.746 | 0.6632 |
| single_B6gs_gapscene | 0.6264 | 0.5819 | **0.6033** | 0.561–0.642 | [0.664, 0.324, 0.555, 0.784] | 0.692 | 0.6085 |
| single_C6_v2_ft | 0.6433 | 0.5364 | **0.5850** | 0.547–0.621 | [0.611, 0.43, 0.388, 0.717] | 0.606 | 0.5970 |
| single_C_v3lab_dinov3_s0 | 0.6560 | 0.5320 | **0.5876** | 0.550–0.620 | [0.635, 0.395, 0.379, 0.719] | 0.446 | 0.5672 |
| single_C_v2lab_s0 | 0.6767 | 0.5085 | **0.5807** | 0.544–0.616 | [0.613, 0.383, 0.326, 0.711] | 0.519 | 0.5782 |
| single_F6_satlas_unet | 0.5826 | 0.4791 | **0.5258** | 0.494–0.558 | [0.684, 0.209, 0.284, 0.739] | 0.425 | 0.5370 |

## Members
| member | seeds | S per-seed mean (recomputed) | S in results.md | S seed-averaged probs (what the ensemble reads) |
|---|---|---|---|---|
| B6gs_gapscene | 1 | 0.6033 | 0.6033 | 0.6033 |
| C6_v2_ft | 1 | 0.5844 | 0.5844 | 0.5844 |
| C_v3lab_dinov3_s0 | 1 | 0.5858 | 0.5858 | 0.5858 |
| C_v2lab_s0 | 1 | 0.5734 | 0.5734 | 0.5734 |
| F6_satlas_unet | 1 | 0.5246 | 0.5246 | 0.5246 |

## Leak checks: 50 splits asserted disjoint
| member | protocol | early-stopping / checkpoint selection source |
|---|---|---|
| B6gs_gapscene | in | 20% train blocks |
| B6gs_gapscene | loso | val site=agadez, val site=aleppo, val site=derna |
| C6_v2_ft | in | 20% train blocks |
| C6_v2_ft | loso | val site=agadez, val site=aleppo, val site=derna |
| C_v3lab_dinov3_s0 | in | 20% train blocks |
| C_v3lab_dinov3_s0 | loso | val site=agadez, val site=aleppo, val site=derna |
| C_v2lab_s0 | in | 20% train blocks |
| C_v2lab_s0 | loso | val site=agadez, val site=aleppo, val site=derna |
| F6_satlas_unet | in | 20% train blocks |
| F6_satlas_unet | loso | val site=agadez, val site=aleppo, val site=derna |
