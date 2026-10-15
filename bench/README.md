# Temporary measurements for Bayaya Platinum

Pictures with known text (`ocr/images`, ground truth in `ocr/gt`) for comparing Bayaya Platinum's text
recognition with Offline Translator's engine ([translator-rs](https://github.com/DavidVentura/translator-rs), MIT).
Texts: sentences from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), English
sentences written for this test, and public-domain prose (Pushkin, «Капитанская дочка», 1836;
Jerome K. Jerome, "Three Men in a Boat", 1889). Everything here is removed after the study.

## gemma/

Processor time of Gemma 4 E2B on the app's tasks for a forwarded voice message, by thread count, next to
GigaAM v3 on the same runner (workflow `bench-gemma.yml`, results in the job log). `messages.json` holds
GigaAM v3 transcripts of three ~90 s messages put together from FLEURS ru test sentences (Google,
CC BY 4.0); `message1.flac` is the first of them.
