# OpenCode Workspace Pack

Muc dich: dong goi context + skills (+ agent profiles) cho OpenCode khi lam viec trong repo nay.

## Cau truc

- `manifest.yaml`: diem vao chinh, liet ke context/skills/agents
- `context/`: tri thuc nen cua project
- `skills/`: quy trinh tac vu theo bai toan
- `agents/`: profile agent chuyen biet (tuy chon)

## Ghi chu

- Tat ca noi dung trong thu muc nay la tai lieu huong dan cho agent, khong anh huong runtime API.
- API contract (`/health`, `/predict/file`, `/predict/base64`) duoc xem la bat bien tru khi co yeu cau ro rang.
