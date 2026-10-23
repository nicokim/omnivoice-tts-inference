# Changelog

## 0.1.0 (2026-10-23)


### Features

* FlashInfer vendorizado, sin grafos ni estado global ([78037f6](https://github.com/nicokim/omnivoice-tts-inference/commit/78037f60322d91b7aa2629c5445f7a5b630899d9))
* generate_streaming por oración con normalize y duration por oración ([663a43a](https://github.com/nicokim/omnivoice-tts-inference/commit/663a43aff9f7e1b8a592c0c87af5390e56f379fe))
* generator= privado en generate() ([6834a1b](https://github.com/nicokim/omnivoice-tts-inference/commit/6834a1bb16966f4dbea507198a328f2b20ed1d51))
* loader propio con codec solo decodificador y kwargs del Hub ([586a210](https://github.com/nicokim/omnivoice-tts-inference/commit/586a210212d36e3a6259328e9abe3bdff8fb4dfe))
* modo int8 por canal con escala en fp32 ([bba6e71](https://github.com/nicokim/omnivoice-tts-inference/commit/bba6e71a4c5b862bdcd8860463fb361b8d95f29f))
* normalización de texto en español (NeMo más las reglas de indextts25) ([a71eaad](https://github.com/nicokim/omnivoice-tts-inference/commit/a71eaada089c62c4572060fa38ad33c29f607eba))


### Bug Fixes

* audio_chunk_duration &lt;= 0 no divide el texto ([576072b](https://github.com/nicokim/omnivoice-tts-inference/commit/576072b114103af6aceb169bbd5b64e44e6e1552))
* el benchmark no cuenta el import de FlashInfer como carga ([a6b6b30](https://github.com/nicokim/omnivoice-tts-inference/commit/a6b6b3087175af525c77ff9c92bc71f814dcf097))
* forward exige attention_mask y no arma caché ([891c78c](https://github.com/nicokim/omnivoice-tts-inference/commit/891c78c14cf853ad42a0323742c447336233a503))
* la caché de TN solo carga archivos regulares y un FIFO no la bloquea ([c21bcfe](https://github.com/nicokim/omnivoice-tts-inference/commit/c21bcfe54935bfc153addaad9d838c9460f34187))
* los kwargs pisan generation_config y un campo desconocido falla ([599744a](https://github.com/nicokim/omnivoice-tts-inference/commit/599744a366321df34a3aa11df432e3647c9dc424))
* los saltos de línea pasan a ser un espacio ([138e87c](https://github.com/nicokim/omnivoice-tts-inference/commit/138e87c16453ea2bb16fbc7f03f61ba8d3f78aa9))
* los trozos siguientes no aplican speed dos veces ([97c622d](https://github.com/nicokim/omnivoice-tts-inference/commit/97c622d7893bfd550c80e73fb02e002342084e72))
* un clip casi mudo devuelve audio vacío ([af196a7](https://github.com/nicokim/omnivoice-tts-inference/commit/af196a7495f4b4b0666a7c0f41ed23619ed48c7b))
* un idioma desconocido falla y los IDs no distinguen mayúsculas ([05cc932](https://github.com/nicokim/omnivoice-tts-inference/commit/05cc932ac7003029ce86dc57e17c5488e0ec2491))
* un texto sin nada que decir falla ([6a5f48b](https://github.com/nicokim/omnivoice-tts-inference/commit/6a5f48b46b339b474560f009c88d084f3a4d4ca1))


### Documentation

* anotar en la spec el resultado de la escucha a ciegas ([8d8ac26](https://github.com/nicokim/omnivoice-tts-inference/commit/8d8ac26795e0b55d3d71cf3f03cc296a1c602239))
* cerrar en la spec las escuchas del stream y del normalizador ([90591cc](https://github.com/nicokim/omnivoice-tts-inference/commit/90591ccbbb3a567a62e1cf03b33a3f7d5f3c884d))
* cifras con el asignador por defecto y aclaraciones de la revisión final ([2f3e18c](https://github.com/nicokim/omnivoice-tts-inference/commit/2f3e18cb15673b48ac75b3040639411fd5907f1b))
* corregir la descripción de postprocess_output ([47dc627](https://github.com/nicokim/omnivoice-tts-inference/commit/47dc6275060fe70d18fac097d096225c3daa522e))
* describir bien lo que hace postprocess_output ([79b6d14](https://github.com/nicokim/omnivoice-tts-inference/commit/79b6d146a6c9be52d54bedbc5245aba188859513))
* incorporar a la spec los resultados de bench2 ([0651aa0](https://github.com/nicokim/omnivoice-tts-inference/commit/0651aa070605320d49f836c796c93af1c7a32bfe))
* medidas de la librería en GPU ([292eea2](https://github.com/nicokim/omnivoice-tts-inference/commit/292eea2e517eea63837fb9cc58e976f434c5042f))
* plan de implementación de v0.1 ([ab516c2](https://github.com/nicokim/omnivoice-tts-inference/commit/ab516c2f8d258c5db35701c03a6d0ef4650a839f))
* README de la librería, docs recortados y script de benchmark ([929573c](https://github.com/nicokim/omnivoice-tts-inference/commit/929573cd9ebc50cb781858cd56be4b58038f3113))
* spec de diseño de omnivoice-tts-inference v0.1 ([78a344f](https://github.com/nicokim/omnivoice-tts-inference/commit/78a344fe67a68d7621d1257b26e6dbfbb5c921c7))
