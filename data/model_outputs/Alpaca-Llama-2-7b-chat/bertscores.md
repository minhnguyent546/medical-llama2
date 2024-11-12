# BERTSCores for Alpaca-Llama-2-7b-chat

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
      <td style="text-align: center;">0.292</td>
      <td style="text-align: center;">0.233</td>
      <td style="text-align: center;">0.263</td>
    </tr>
    <tr>
      <td>gpt-3.5-turbo</td>
      <td style="text-align: center;">0.328</td>
      <td style="text-align: center;">0.280</td>
      <td style="text-align: center;">0.304</td>
    </tr>
    <tr>
      <td>text-davinci-003</td>
      <td style="text-align: center;">0.206</td>
      <td style="text-align: center;">0.409</td>
      <td style="text-align: center;">0.306</td>
    </tr>
    <tr>
      <td>claude-2</td>
      <td style="text-align: center;">0.258</td>
      <td style="text-align: center;">0.179</td>
      <td style="text-align: center;">0.219</td>
    </tr>
    <tr>
      <td rowspan="4">iCliniq-1k</td>
      <td>gpt-4</td>
      <td style="text-align: center;">0.228</td>
      <td style="text-align: center;">0.168</td>
      <td style="text-align: center;">0.198</td>
    </tr>
    <tr>
      <td>gpt-3.5-turbo</td>
      <td style="text-align: center;">0.277</td>
      <td style="text-align: center;">0.260</td>
      <td style="text-align: center;">0.269</td>
    </tr>
    <tr>
      <td>text-davinci-003</td>
      <td style="text-align: center;">0.152</td>
      <td style="text-align: center;">0.362</td>
      <td style="text-align: center;">0.256</td>
    </tr>
    <tr>
      <td>claude-2</td>
      <td style="text-align: center;">0.181</td>
      <td style="text-align: center;">0.129</td>
      <td style="text-align: center;">0.155</td>
    </tr>
  </tbody>
</table>

BERTScore hash codes:

- Scaled: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.42.3)-rescaled_fast-tokenizer
- Un-scaled: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.42.3)_fast-tokenizer
