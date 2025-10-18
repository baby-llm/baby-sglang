root@autodl-container-f75c4aac4c-3fec72ae:~/baby-sglang# python run_demo.py --comprehensive --seed=42
🚀 Starting comprehensive Engine smoke tests...

config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 684/684 [00:00<00:00, 4.01MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors:   1%|▊                                                                                                                                              | 17.3M/3.09G [00:07<19:59, 2.56MB/s]  2025-10-18T03:08:39.838275Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e13dde02182e6617a7fc77472e9339d1ad1a4af0053da4da068425205c0432e?X-Xet-Signed-Range=bytes%3D0-58567240&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTEzZGRlMDIxODJlNjYxN2E3ZmM3NzQ3MmU5MzM5ZDFhZDFhNGFmMDA1M2RhNGRhMDY4NDI1MjA1YzA0MzJlP1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NjcyNDAmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=dFkc3Hm9qfkjW~iO3YoO~R9qnMjXEzfMsTYSCUFEPFRyhBhAFoVtwDPdfDTRSZoh7JcPNveQ~3O-065P7grqZICZOUIzC8QiV1UUeGJY79s7hl4-wF4fLeIhIQl6PoW0Vuzh8U4HaaoCGdGoudo7oMqfTWp1xQUBhUzHevulBgaAus22gn8-C0wuZvfn-tdY3qglrYNTj~2bOAtDasUouHJqJm-dX0awNgk5AnFFDrD4InYXAEdDebcypGMPJ2pyVhVpLh4vQXWICbuGyxoSCYL9p-MKqS72ZRQfJgVZco4wqxjySIxQNl5~~3vOpZ~FJAsJ2OUfdFCr7kc-gFoOLQ__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:39.838323Z  WARN  Retry attempt #0. Sleeping 2.757935811s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

model.safetensors:   1%|▊                                                                                                                                              | 17.9M/3.09G [00:09<27:08, 1.88MB/s]  2025-10-18T03:08:41.626612Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/123ebd9cff8f3325e868d95da33ddabdccd988a793b0587c66a87af4afa14c86?X-Xet-Signed-Range=bytes%3D0-58445505&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8xMjNlYmQ5Y2ZmOGYzMzI1ZTg2OGQ5NWRhMzNkZGFiZGNjZDk4OGE3OTNiMDU4N2M2NmE4N2FmNGFmYTE0Yzg2P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg0NDU1MDUmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=mN6jghWKssfNgqHSrHLS~5itWQDkpOWuZdru3KwBZhcIggXwqHCcbSn7CA3-6e6rg2InQhUcWa6KCFklGKF0ywoikmDT7lg3TUhql372F9TuM0cMpSP85Wn38mKJFHzlFM0bsKLmiliUDeBk3Pc8V0ZWmYT7spy-5P7VkeKNgAliWZoDVs7Q0qqMt55MTFvOaBTWtHAtE9mogHwObR5xfEXTCd5-CXxHKuMsLfHsee58NILbbW3IX5OC-TIYdEttlTQe6CedrBdfm~JqvXzq6jtOPjd1UNK-O1et4RsZLcd0BObcJ-Mix-e~oDmKURGzTw3-OsxEGjX5pygc5lVSag__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:41.626676Z  WARN  Retry attempt #0. Sleeping 824.717123ms before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

model.safetensors:   1%|▊                                                                                                                                              | 18.6M/3.09G [00:09<25:34, 2.00MB/s]  2025-10-18T03:08:41.731468Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e62a2df42ac0e49ec2ba186798bfbd85fad68156a22d1a1671fa4252e538298?X-Xet-Signed-Range=bytes%3D0-58543148&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTYyYTJkZjQyYWMwZTQ5ZWMyYmExODY3OThiZmJkODVmYWQ2ODE1NmEyMmQxYTE2NzFmYTQyNTJlNTM4Mjk4P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NDMxNDgmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=fSUPvVy2qJubcsoweihGyCYIBtYqjhFHB7BjnpBrG-mGqm0RhcZ4pFKMxKthEpuZQY6R3Wf6yrWNoVq5ODDfyf-vjA4HeonrM6z5jv7DjQ1lR2H~rGyuFAuzEUCIwBMfGW-HcTVeOtFhQoYuJmGU9CUEj8SJeerp1IVWBFduxcMqBJhzFZTeViffIUsLDoWsOWI8zWv~kwWPtaLtdJymyqpLstAykpy3cFGiUFH3crQATJRjRybWEQOc6cTFG32fSEPa7PMKDx-Hhoa0Cr31WCo2gjbHL~tIyqY0wt5-ECx8MOQeOVYDidarAHZHAFK2IPCk7ELjgpDHllG312VJow__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:41.731503Z  WARN  Retry attempt #0. Sleeping 183.824187ms before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

  2025-10-18T03:08:41.917362Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e62a2df42ac0e49ec2ba186798bfbd85fad68156a22d1a1671fa4252e538298?X-Xet-Signed-Range=bytes%3D0-58543148&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTYyYTJkZjQyYWMwZTQ5ZWMyYmExODY3OThiZmJkODVmYWQ2ODE1NmEyMmQxYTE2NzFmYTQyNTJlNTM4Mjk4P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NDMxNDgmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=fSUPvVy2qJubcsoweihGyCYIBtYqjhFHB7BjnpBrG-mGqm0RhcZ4pFKMxKthEpuZQY6R3Wf6yrWNoVq5ODDfyf-vjA4HeonrM6z5jv7DjQ1lR2H~rGyuFAuzEUCIwBMfGW-HcTVeOtFhQoYuJmGU9CUEj8SJeerp1IVWBFduxcMqBJhzFZTeViffIUsLDoWsOWI8zWv~kwWPtaLtdJymyqpLstAykpy3cFGiUFH3crQATJRjRybWEQOc6cTFG32fSEPa7PMKDx-Hhoa0Cr31WCo2gjbHL~tIyqY0wt5-ECx8MOQeOVYDidarAHZHAFK2IPCk7ELjgpDHllG312VJow__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:41.917422Z  WARN  Retry attempt #1. Sleeping 2.056040429s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

  2025-10-18T03:08:42.452956Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/123ebd9cff8f3325e868d95da33ddabdccd988a793b0587c66a87af4afa14c86?X-Xet-Signed-Range=bytes%3D0-58445505&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8xMjNlYmQ5Y2ZmOGYzMzI1ZTg2OGQ5NWRhMzNkZGFiZGNjZDk4OGE3OTNiMDU4N2M2NmE4N2FmNGFmYTE0Yzg2P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg0NDU1MDUmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=mN6jghWKssfNgqHSrHLS~5itWQDkpOWuZdru3KwBZhcIggXwqHCcbSn7CA3-6e6rg2InQhUcWa6KCFklGKF0ywoikmDT7lg3TUhql372F9TuM0cMpSP85Wn38mKJFHzlFM0bsKLmiliUDeBk3Pc8V0ZWmYT7spy-5P7VkeKNgAliWZoDVs7Q0qqMt55MTFvOaBTWtHAtE9mogHwObR5xfEXTCd5-CXxHKuMsLfHsee58NILbbW3IX5OC-TIYdEttlTQe6CedrBdfm~JqvXzq6jtOPjd1UNK-O1et4RsZLcd0BObcJ-Mix-e~oDmKURGzTw3-OsxEGjX5pygc5lVSag__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:42.453011Z  WARN  Retry attempt #1. Sleeping 4.533650454s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

  2025-10-18T03:08:42.598277Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e13dde02182e6617a7fc77472e9339d1ad1a4af0053da4da068425205c0432e?X-Xet-Signed-Range=bytes%3D0-58567240&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTEzZGRlMDIxODJlNjYxN2E3ZmM3NzQ3MmU5MzM5ZDFhZDFhNGFmMDA1M2RhNGRhMDY4NDI1MjA1YzA0MzJlP1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NjcyNDAmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=dFkc3Hm9qfkjW~iO3YoO~R9qnMjXEzfMsTYSCUFEPFRyhBhAFoVtwDPdfDTRSZoh7JcPNveQ~3O-065P7grqZICZOUIzC8QiV1UUeGJY79s7hl4-wF4fLeIhIQl6PoW0Vuzh8U4HaaoCGdGoudo7oMqfTWp1xQUBhUzHevulBgaAus22gn8-C0wuZvfn-tdY3qglrYNTj~2bOAtDasUouHJqJm-dX0awNgk5AnFFDrD4InYXAEdDebcypGMPJ2pyVhVpLh4vQXWICbuGyxoSCYL9p-MKqS72ZRQfJgVZco4wqxjySIxQNl5~~3vOpZ~FJAsJ2OUfdFCr7kc-gFoOLQ__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:42.598339Z  WARN  Retry attempt #1. Sleeping 3.396796461s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

  2025-10-18T03:08:43.975564Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e62a2df42ac0e49ec2ba186798bfbd85fad68156a22d1a1671fa4252e538298?X-Xet-Signed-Range=bytes%3D0-58543148&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTYyYTJkZjQyYWMwZTQ5ZWMyYmExODY3OThiZmJkODVmYWQ2ODE1NmEyMmQxYTE2NzFmYTQyNTJlNTM4Mjk4P1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NDMxNDgmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=fSUPvVy2qJubcsoweihGyCYIBtYqjhFHB7BjnpBrG-mGqm0RhcZ4pFKMxKthEpuZQY6R3Wf6yrWNoVq5ODDfyf-vjA4HeonrM6z5jv7DjQ1lR2H~rGyuFAuzEUCIwBMfGW-HcTVeOtFhQoYuJmGU9CUEj8SJeerp1IVWBFduxcMqBJhzFZTeViffIUsLDoWsOWI8zWv~kwWPtaLtdJymyqpLstAykpy3cFGiUFH3crQATJRjRybWEQOc6cTFG32fSEPa7PMKDx-Hhoa0Cr31WCo2gjbHL~tIyqY0wt5-ECx8MOQeOVYDidarAHZHAFK2IPCk7ELjgpDHllG312VJow__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:43.975632Z  WARN  Retry attempt #2. Sleeping 11.251173401s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

  2025-10-18T03:08:46.027332Z  WARN  Reqwest(reqwest::Error { kind: Request, url: "<https://transfer.xethub.hf.co/xorbs/default/0e13dde02182e6617a7fc77472e9339d1ad1a4af0053da4da068425205c0432e?X-Xet-Signed-Range=bytes%3D0-58567240&X-Xet-Session-Id=01K7TMMM6H06EMWWH0NVVYW18H&Expires=1760760516&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly90cmFuc2Zlci54ZXRodWIuaGYuY28veG9yYnMvZGVmYXVsdC8wZTEzZGRlMDIxODJlNjYxN2E3ZmM3NzQ3MmU5MzM5ZDFhZDFhNGFmMDA1M2RhNGRhMDY4NDI1MjA1YzA0MzJlP1gtWGV0LVNpZ25lZC1SYW5nZT1ieXRlcyUzRDAtNTg1NjcyNDAmWC1YZXQtU2Vzc2lvbi1JZD0wMUs3VE1NTTZIMDZFTVdXSDBOVlZZVzE4SCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDc2MDUxNn19fV19&Signature=dFkc3Hm9qfkjW~iO3YoO~R9qnMjXEzfMsTYSCUFEPFRyhBhAFoVtwDPdfDTRSZoh7JcPNveQ~3O-065P7grqZICZOUIzC8QiV1UUeGJY79s7hl4-wF4fLeIhIQl6PoW0Vuzh8U4HaaoCGdGoudo7oMqfTWp1xQUBhUzHevulBgaAus22gn8-C0wuZvfn-tdY3qglrYNTj~2bOAtDasUouHJqJm-dX0awNgk5AnFFDrD4InYXAEdDebcypGMPJ2pyVhVpLh4vQXWICbuGyxoSCYL9p-MKqS72ZRQfJgVZco4wqxjySIxQNl5~~3vOpZ~FJAsJ2OUfdFCr7kc-gFoOLQ__&Key-Pair-Id=K2L8F4GPSG1IFC>", source: hyper_util::client::legacy::Error(Connect, Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })) }). Retrying...
    at /home/runner/work/xet-core/xet-core/cas_client/src/http_client.rs:233

  2025-10-18T03:08:46.027405Z  WARN  Retry attempt #2. Sleeping 7.382803547s before the next attempt
    at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/reqwest-retry-0.7.0/src/middleware.rs:171

model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.09G/3.09G [01:57<00:00, 26.3MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:00<00:00, 380kB/s]
tokenizer_config.json: 7.23kB [00:00, 10.7MB/s]
vocab.json: 2.78MB [00:01, 2.37MB/s]
merges.txt: 1.67MB [00:00, 18.7MB/s]
tokenizer.json: 7.03MB [00:00, 24.1MB/s]
============ Single Prompt Test ============
Prompt: Hello, how are you?
Output: 'AppState of the world, and the state of the world, and the state of'
✅ Single prompt test PASSED
=============================================

✅ Empty input test PASSED
Initializing Engine with model: Qwen/qwen2.5-1.5B
============ Engine Smoke Test ============
Testing with 4 prompts
Sampling params: max_new_tokens=24, do_sample=False
[0] Prompt: Summarize the key benefits of unit testing in software engineering.
    Output: " 's\nB. engineering in the test of 's\nC. engineering in the test of\nD. engineering"

[1] Prompt: Explain the concept of attention in transformer models.
    Output: " 's\nB. in the models of 's\nC. in the models of\nD. in the models"

[2] Prompt: Draft a short professional email requesting a project status update.
    Output: " 's status\n    - update a project's status\n    - update a project's status\n    - update a"

[3] Prompt: Give three tips to improve public speaking.
    Output: " 's\nB. speaking to the public\nC. speaking to the public 's\nD. speaking to the"

✅ Engine smoke test PASSED
=============================================

Initializing Engine with model: Qwen/qwen2.5-1.5B
============ Engine Smoke Test ============
Testing with 4 prompts
Sampling params: max_new_tokens=20, do_sample=False
[0] Prompt: Hello, introduce yourself briefly.
    Output: " 's\nAnswer:\n\nAssistant: B\n\nHuman: The following is a single-choice question from a"

[1] Prompt: 用中文介绍一下量子计算的基本原理。
    Output: ' hé bù\n\nAssistant: 量子计算;量子计算;量子计算;量子计算;量子'

[2] Prompt: Write a Python function to compute Fibonacci numbers efficiently.
    Output: " 's\n    # 1. The function should take a single argument, n, which represents"

[3] Prompt: 给出三条关于上海旅游的建议。
    Output: ' hé jià 旅游的建议是关于旅游的建议。'

✅ Engine smoke test PASSED
=============================================

Initializing Engine with model: Qwen/qwen2.5-1.5B
============ Engine Smoke Test ============
Testing with 4 prompts
Sampling params: max_new_tokens=16, do_sample=True
                temperature=0.7, top_k=20, top_p=0.9
[0] Prompt: 用中文简要介绍大语言模型的工作原理。
    Output: ' hé bù 要求\n\nHuman: 2020年'

[1] Prompt: 请解释A/B测试的基本流程和注意事项。
    Output: ' hé\nA. 正确\nB. 错误\n答案:\n'

[2] Prompt: 写一段100字以内的自我介绍，语气自然。
    Output: ' hé0内0,2000字内0,000'

[3] Prompt: 给出三条提高学习效率的建议。
    Output: ' hé shu\n\nAssistant: 效率\n\nHuman: 在下列选项中'

✅ Engine smoke test PASSED
=============================================

🎉 All smoke tests completed successfully!
