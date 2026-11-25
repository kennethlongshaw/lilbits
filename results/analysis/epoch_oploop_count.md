# epoch_oploop_count

- task: count
- **ops**
  - score: 0.375
  - expected_chain_rate: 1
  - router_entropy_by_loop: 0.00131, 0.000245, 0.0129, 0.0264
  - ops_when_right: {'filter': 15}, {'count': 15}, {'count': 15}, {'count': 15}
  - ops_when_wrong: {'filter': 25}, {'count': 25}, {'count': 25}, {'count': 25}
- **attend**
  - score: 0.375
  - expected_chain_rate: 0
  - router_entropy_by_loop: nan, nan, nan, nan
  - ops_when_right: {'attend': 15}, {'attend': 15}, {'attend': 15}, {'attend': 15}
  - ops_when_wrong: {'attend': 25}, {'attend': 25}, {'attend': 25}, {'attend': 25}
