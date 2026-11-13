# Языковые веса Tesseract для выборочного восстановления названий

Рабочее место моделей: `models/tesseract/eng.traineddata` и
`models/tesseract/rus.traineddata`. Это языковые веса внешнего Tesseract OCR,
а не Python-библиотеки. Исполняемый файл Tesseract 5.x устанавливается отдельно.
Python-зависимости уже перечислены в `requirements-web.txt`.

Установка весов из официального `tesseract-ocr/tessdata_fast`, версия `4.1.0`,
от корня проекта:

```powershell
New-Item -ItemType Directory -Path models/tesseract -Force
Invoke-WebRequest https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/rus.traineddata -OutFile models/tesseract/rus.traineddata
Invoke-WebRequest https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata -OutFile models/tesseract/eng.traineddata
```

На проверенном рабочем месте `rus` скачан из указанного источника, `eng`
скопирован из установленного Tesseract. SHA-256 установленных файлов:

| Файл | SHA-256 |
|---|---|
| `rus.traineddata` | `e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225` |
| `eng.traineddata` | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |

Если оба запрошенных языка имеются в этой папке, модуль передаёт её Tesseract
через `--tessdata-dir`. Иначе используется системная установка. Отсутствие
движка или нужного языка фиксируется в диагностике; замена имени без второго
движка не подтверждается. Во время обработки чеков ничего не скачивается.
Файлы весов исключены из Git. `configs/tsv` — стандартный текстовый профиль
вывода TSV для диагностических утилит Tesseract.
