# Coverage of the simplified open labels

The source workbook has one visible sheet, `labels`, with four columns, 42 data rows and no cell comments. Column C contains 22 positive and 20 negative rows. Column D contains 14 sparse possible-category hints across seven IDs; three hints occur on explicitly negative rows. Eleven positive rows have no category hint. These are row counts: duplicate source footage reduces the corpus to 40 unique byte-hash groups; recording independence is unknown, as recorded in the fine-tuning audit.

The revised architecture provides **7/7 hinted types with visual detection paths**, within nine registered visual scenarios. The separate legal registry retains **three automatic legal rules**. Pipeline 5.0 screens all nine scenarios and verifies visible review concerns using source-linked images. A supported visual concern can reach human review while ownership, authorization, internal policy or transaction lineage remains unresolved. These counts describe registered architecture coverage, not measured all-case recall. The reproducible aggregate report is `docs/label-coverage-v5.json`.

The initial architecture could automatically emit only three event types. Four hinted scenario families were manual catalog entries with no candidate-verification path, and one category was missing from the catalog. The following table records that initial gap and the predicates the revised screening path must inspect. A detector examining a scenario does not guarantee a correct detection.

| Workbook category | Positive / negative rows | Initial gap | Required visible observations and unresolved facts |
| --- | ---: | --- | --- |
| `previous_transaction_cash_exposed` | 4 / 0 | Automatic rule existed, but local windows could miss cash lineage across the transaction transition. | Prior cash acceptance, source-linked cash continuity, storage relationship and a distinct next transaction. Multiple unidentified piles alone are insufficient. |
| `previous_cash_stored` | 1 / 0 | Review-only procedure counterpart had no event finding path. | Track storage placement and later exposure; share the cash-storage family with the event counterpart to avoid duplicate breaches. |
| `cash_intake_without_client_presence` | 1 / 0 | Automatic rule existed, but intake/cash lineage was often unresolved. | Observe acceptance/counting together with the relevant client zone and establish actual absence. A client outside the image is unknown. |
| `backoffice_processing_control` | 1 / 3 | Manual catalog only. | Observe non-client counting, sorting or packing and visible oversight. Ordinary client intake/payout is excepted. Official supervisory roles and off-camera oversight remain unknown without records. Negative hinted examples must stay negative. |
| `vault_cash_segregation_storage` | 1 / 0 | Exact category absent from both catalog and registry. | Observe the storage scene, cash containers/compartments, placement or mixing and temporal continuity. Cash category, segregation requirements and designated vault status require explicit evidence; do not invent a legal clause from the label ID. |
| `keys_stamps_secure_handling` | 2 / 0 | Manual catalog only; small-object localization was incomplete. | Observe key/stamp/code-device handling, placement, transfer or unattended exposure with detailed source frames. A precise breach requires the actual internal policy and authorization facts. |
| `personal_belongings_in_cash_unit` | 1 / 0 | Manual catalog only; personal-item sightings never reached rule-specific verification. | Observe the item, placement/entry and applicable cash-unit area. Ownership, work purpose and medicine/communication-device exceptions remain explicit unknowns when unavailable. |

The revision screens the whole clip for these scenario families, retains candidate time spans and detailed views, then asks predicate-specific questions over a wider temporal context. Unknown legal or policy facts produce an evidence-linked review item. `previous_cash_stored` remains an explicit scenario alias of `previous_transaction_cash_exposed`, preserving both hinted IDs and recording their relationship without treating them as independent accuracy examples. They retain separately typed candidates and verification because a storage cue and a transition with prior-client cash require different evidence. The formerly missing vault category is an unmapped review scenario with no invented legal citation.

`scripts/audit_label_coverage.py` independently compares the visual scenario registry and automatic legal registry/catalog with the immutable workbook. It reports their coverage separately and exports aggregate category counts only. It is outside the inference dependency path and does not export case names, timestamps or media hashes. Run:

```powershell
.\.venv\Scripts\python.exe scripts/audit_label_coverage.py --output docs/label-coverage-v5.json
```

The seven hinted IDs do not explain the eleven positive rows lacking categories. Detecting every labeled positive still requires visual evaluation across all source groups and reviewing their event types. No all-case recall claim follows from catalog coverage.
