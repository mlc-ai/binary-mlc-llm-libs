# binary-mlc-llm-libs

Model libraries are stored in the format:

```
{model_name}/{model_name}-{quantization}-{metadata}-{platform}.{suffix}
```

Metadata: 
- `ctx`: context window size
- `sw`: sliding window size
- `cs`: prefill chunk size

For default configurations of metadata, we do not include that in the file name. We also do not include prefill chunk size if it is the same as the context window size or sliding window size (the default choice).

### Default Metadata

<table style="width:100%">
  <thead>
    <tr>
      <th style="width:25%"> </th>
      <th style="width:20%">Context Window Size</th>
      <th style="width:20%">Sliding Window Size</th>
      <th style="width:20%">Prefill Chunk Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Llama-2-7b-chat-hf</td>
      <td>4096</td>
      <td>N/A</td>
      <td>4096</td>
    </tr>
    <tr>
      <td>Llama-2-13b-chat-hf</td>
      <td>4096</td>
      <td>N/A</td>
      <td>4096</td>
    </tr>
    <tr>
      <td>Llama-2-70b-chat-hf</td>
      <td>4096</td>
      <td>N/A</td>
      <td>4096</td>
    </tr>
    <tr>
      <td>Mistral-7B-Instruct-v0.2</td>
      <td>N/A</td>
      <td>4096</td>
      <td>4096</td>
    </tr>
    <tr>
      <td>RedPajama-INCITE-Chat-3B-v1</td>
      <td>2048</td>
      <td>N/A</td>
      <td>2048</td>
    </tr>
    <tr>
      <td>phi-2</td>
      <td>2048</td>
      <td>N/A</td>
      <td>2048</td>
    </tr>
    <tr>
      <td>phi-1_5</td>
      <td>2048</td>
      <td>N/A</td>
      <td>2048</td>
    </tr>
    <tr>
      <td>gpt2</td>
      <td>1024</td>
      <td>N/A</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>gpt2-medium</td>
      <td>1024</td>
      <td>N/A</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
