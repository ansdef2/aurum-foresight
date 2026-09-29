# Развёртывание Aurum Foresight System (сервер + HTTPS)

Сервер: Ubuntu 24.04, 2 vCPU / 4 ГБ, российский регион (GigaChat и российские СМИ доступны без обходных путей).
Сайт не тянет ничего с внешних CDN: шрифт Montserrat, React и логотип лежат в `app/static/`.

**Timeweb Cloud.** «Облачные серверы» → «Создать»: Ubuntu 24.04 (можно образ из маркетплейса с Docker —
тогда шаг 2 с установкой Docker пропустить), публичный IPv4, свой SSH-ключ. Если к серверу подключён облачный
firewall — откройте входящие 22, 80, 443. A-запись заводится в разделе «Домены» → DNS.
Если `docker compose up` не может скачать образы `postgres`, `caddy`, `python` с Docker Hub (403 / timeout),
подключите зеркало Docker Hub от Timeweb Cloud (адрес — в их документации) в `/etc/docker/daemon.json`:
`{"registry-mirrors": ["https://<зеркало>"]}`, затем `systemctl restart docker`.

1. DNS: A-запись поддомена (например `aurum.<ваш-домен>`) -> IP сервера. Проверка: `dig +short aurum.<домен>`.
2. Docker и файрвол:
   ```bash
   curl -fsSL https://get.docker.com | sh
   ufw allow 22,80,443/tcp && ufw --force enable
   ```
3. Код — из архива или из GitHub (что-то одно):
   ```bash
   # архив: с вашего компьютера  scp aurum-foresight-system.tar.gz root@IP_СЕРВЕРА:~/
   mkdir -p /opt/ws && cd /opt/ws && tar xzf ~/aurum-foresight-system.tar.gz --strip-components=1
   # или GitHub:
   # git clone https://github.com/ansdef2/aurum-foresight.git /opt/ws && cd /opt/ws
   cp .env.example .env && nano .env
   ```
   Заполнить: `WS_DOMAIN`, `GIGACHAT_AUTH_KEY`, `GITHUB_TOKEN` (без токена GitHub Search даёт 10 запросов в минуту — обучение будет очень долгим), `CONTACT_EMAIL`.
4. Сертификат Минцифры для GigaChat:
   ```bash
   curl -o certs/russian_trusted_root_ca.pem https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
   ```
   Если формат не PEM: `openssl x509 -inform DER -in certs/russian_trusted_root_ca.pem -out certs/russian_trusted_root_ca.pem`.
   Экстренно (только для отладки): `GIGACHAT_CA_BUNDLE=` и `GIGACHAT_VERIFY_SSL=false`.
5. Старт: `docker compose up -d --build`, проверка: `curl https://<домен>/api/health` и сайт `https://<домен>/`.
   Проверка ключа и идентификаторов моделей GigaChat (печатает список моделей, доступных ключу, и помечает
   неверные `MODEL_*` строкой `FAIL`):
   `docker compose --profile tools run --rm tools python -m scripts.smoke_sources`
6. Детектор уже обучен: `data/model/detector.json` в архиве. `/api/health` должен показать `"detector": "trained"`.
   Переобучать НЕ нужно. По желанию — сквозная оценка (метрики слайда 10): `nohup ./scripts/run_eval_e2e.sh > eval.log 2>&1 &`
7. Прогрев кэша: `./scripts/prewarm.sh https://<домен>` (запросы идут по одному, до 17 минут каждый).
8. Файл `data/eval/dataset.xlsx` уже лежит на сервере, но в образ API он не попадает (`.dockerignore`).

Логи: `docker compose logs -f api`, `docker compose logs -f caddy`.

## Типовые сбои

| В логе | Причина | Что сделать |
|---|---|---|
| `GigaChat HTTP 429` в `/api/jobs/{id}/llm` | превышен лимит одновременных запросов ключа | `LLM_MAX_CONCURRENCY=1` в `.env`, затем `docker compose up -d api` |
| `GigaChat HTTP 400` / `404` с текстом про модель | неверный идентификатор модели | взять идентификатор из списка `smoke_sources` |
| `GigaChat OAuth HTTP 401` | неверный ключ или scope | `GIGACHAT_AUTH_KEY`, `GIGACHAT_SCOPE` |
| `Нет файла сертификата GIGACHAT_CA_BUNDLE` / `SSLError` | нет сертификата Минцифры | шаг 4 |
| `ReadTimeout` у модели | долгий ответ Max | `LLM_TIMEOUT_SEC=300`; повторы уже включены (`LLM_RETRIES`) |
| `403 https://api.github.com/...` | лимит поиска GitHub | клиент сам ждёт сброса; проверить `GITHUB_TOKEN` |

Каждая попытка вызова модели, включая неудачные, видна в журнале `GET /api/jobs/{id}/llm`;
если GigaChat ответил не той моделью, что запрошена, в журнале будет «запрошенная → фактическая».

