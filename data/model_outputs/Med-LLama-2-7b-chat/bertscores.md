# BERTSCores for Med-Alpaca-2-7b-chat

<table>
  <thead>
    <tr>
      <th>Test set</th>
      <th>Ref. model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">MedInstruct-test</td>
      <td>gpt-4</td>
      <td style="text-align: center;">0.298</td>
      <td style="text-align: center;">0.260</td>
      <td style="text-align: center;">0.279</td>
    </tr>
    <tr>
      <td>gpt-3.5-turbo</td>
      <td style="text-align: center;">0.365</td>
      <td style="text-align: center;">0.336</td>
      <td style="text-align: center;">0.350</td>
    </tr>
    <tr>
      <td>text-davinci-003</td>
      <td style="text-align: center;">0.167</td>
      <td style="text-align: center;">0.393</td>
      <td style="text-align: center;">0.278</td>
    </tr>
    <tr>
      <td>claude-2</td>
      <td style="text-align: center;">0.248</td>
      <td style="text-align: center;">0.191</td>
      <td style="text-align: center;">0.220</td>
    </tr>
    <tr>
      <td rowspan="4">iCliniq-1k</td>
      <td>gpt-4</td>
      <td style="text-align: center;">0.205</td>
      <td style="text-align: center;">0.199</td>
      <td style="text-align: center;">0.203</td>
    </tr>
    <tr>
      <td>gpt-3.5-turbo</td>
      <td style="text-align: center;">0.267</td>
      <td style="text-align: center;">0.313</td>
      <td style="text-align: center;">0.290</td>
    </tr>
    <tr>
      <td>text-davinci-003</td>
      <td style="text-align: center;">0.089</td>
      <td style="text-align: center;">0.351</td>
      <td style="text-align: center;">0.217</td>
    </tr>
    <tr>
      <td>claude-2</td>
      <td style="text-align: center;">0.158</td>
      <td style="text-align: center;">0.159</td>
      <td style="text-align: center;">0.159</td>
    </tr>
  </tbody>
</table>

BERTScore hash codes:

- Scaled: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.42.3)-rescaled_fast-tokenizer
- Un-scaled: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.42.3)_fast-tokenizer
