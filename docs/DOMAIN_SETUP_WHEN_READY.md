# synapsecore.io — Setup Guide (Execute When Budget Available)

## Step 1 — Register domain (~$15-41/year)
- **Cloudflare Registrar** (cheapest, no markup): cloudflare.com/products/registrar/
- **Namecheap** (alternative): namecheap.com
- Target: `synapsecore.io`

## Step 2 — Email (choose one)

### Option A: Cloudflare Email Routing (FREE)
1. Cloudflare Dashboard → your domain → Email → Email Routing
2. Add route: `allocations@synapsecore.io` → forward to `mister3.ai@gmail.com`
3. Done in 5 minutes. Incoming works, replies show Gmail sender.

### Option B: Google Workspace Business Starter ($6/mo)
1. workspace.google.com → Start free trial
2. Verify domain via DNS TXT record in Cloudflare
3. Create user: `allocations@synapsecore.io`
4. Full send/receive with corporate identity.

## Step 3 — Update Institutional_Memo_Hyperion.html
Find the comment `<!-- DOMAIN READY -->` and replace:
```
mister3.ai@gmail.com
```
with:
```
allocations@synapsecore.io
```

## Step 4 — SPF/DKIM/DMARC (critical for deliverability)
Add DNS TXT records per Google Workspace instructions.
Without these, emails to Family Offices may land in spam.

## Step 5 — Regenerate PDF
Open Institutional_Memo_Hyperion.html in Chrome → Ctrl+P → Save as PDF
Background graphics: ON | Margins: None

---
**Total time once domain is purchased: ~30 minutes**
**Cost: domain ~$15-41/year + optionally $6/mo Workspace**
