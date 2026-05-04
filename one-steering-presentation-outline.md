# One Steering — 12-Minute Presentation Outline
**Contest: The Innovators 2026 (ROCCO)**
**Presented by: TOMIA & MACH**
**Target: 1:1 conversation guide with your manager**

---

## Segment 1 — Opening Hook (0:00 – 1:00 | ~1 min)

**Goal:** Set the stage. Make the problem feel real and urgent.

- Open with the core tension: *"Managing wholesale commercial agreements has never been simple — and today it is a key operational challenge for commercial, finance, and engineering teams."*
- Briefly introduce the joint solution: **One Steering** — a unified approach aligning commercial strategy with real-time network execution.
- Introduce yourselves and company roles (Elisa Bortolussi, Product Manager at MACH; Caetano Pessoa, Head of Marketing at TOMIA).

**Talking point to not forget:** Emphasize this is a *joint collaboration* — not just one company's pitch.

---

## Segment 2 — The Commercial Problem (1:00 – 3:00 | ~2 min)

**Goal:** Show how the market has become structurally more complex on the commercial side.

- The rise of IoT has fundamentally changed how agreements are designed and managed.
- What was once a single commercial agreement has **multiplied by 4 or 5**.
- Multiple coexisting settlement mechanisms: **TAP, IoT Discount, BCE (Billing & Charging Evolution)** — often running on different billing platforms.
- Calculations are handled manually → risk of discrepancies, delayed settlements, slower cash flow.

**Key message:** Complexity has outpaced the tools operators currently use.

---

## Segment 3 — The Network Problem (3:00 – 4:30 | ~1.5 min)

**Goal:** Mirror the commercial problem on the network/engineering side.

- Steering services must replicate every commercial segment into multiple network configurations.
- Segments behave differently: device types vary, access technologies differ, allowed services change, usage patterns diverge.
- Ongoing phase-out of 2G/3G networks forces dynamic, real-time steering.
- Clear correlation: **higher network quality → higher data consumption → more revenue**.
- Steering is not just a technical function — it's a **financial lever**.

**Key message:** When commercial intent and network behavior are misaligned, operators lose money. Managing this manually is *"like finding a needle in a haystack."*

---

## Segment 4 — The Industry Gap (4:30 – 6:00 | ~1.5 min)

**Goal:** Diagnose the current state of the industry and why existing tools fall short.

- Commercial agreements, settlement processes, and steering decisions are managed **in silos**.
- Most operators lack tools that fully reflect growing complexity across segments, charging models, and network configs.
- Automation today is **partial**: focused on agreements OR settlement OR steering — never all three.
- What's missing: **cross-service automation** that connects all steps and eliminates manual calculations.
- The market wants *simplification* — but simplification doesn't mean fewer segments. It means **making complexity manageable and predictable**.

**Key message:** The industry's call for real-time, actionable analytics that close the loop between commercial targets and network execution is unanswered — until now.

---

## Segment 5 — Introducing One Steering (6:00 – 7:30 | ~1.5 min)

**Goal:** Present the solution clearly and confidently.

- **One Steering** bridges the gap between agreement commitments and real-world subscriber steering.
- Two sides of the solution:
  - **MACH – One Agreement:** Automated forecasting, commitment tracking, steering recommendations with estimated discount cost, downloadable or API-provisioned.
  - **TOMIA – IPN (Intelligent Preferred Network):** Segment-specific steering distributions, granular quality monitoring, region/city-level steering rules.
- Together they create a **closed loop**: commercial intent and network behavior finally work as one.

**Key message:** Neither company could deliver this alone. The power is in the combination.

---

## Segment 6 — How It Works in Practice (7:30 – 9:00 | ~1.5 min)

**Goal:** Make it concrete with real-world examples.

- Example 1 (Group commitment): An operator has a group-level commitment across multiple country affiliates. If steering for even one affiliate is miscalculated → commitment penalty. One Steering prevents this.
- Example 2 (Cross-commitment): Commercial teams negotiate cross-commitments spanning retail and IoT → complex accruals and steering. One Steering handles this automatically.
- Example 3 (Granular steering): In the US, Network A is preferred nationwide — but IPN detects Network B has better coverage in Washington → automatically steers subscribers to Network B there, without breaking commercial commitments.

**Key message:** The solution works at scale, with granularity, in real-world messy environments.

---

## Segment 7 — Results & Business Value (9:00 – 10:00 | ~1 min)

**Goal:** Land the financial impact.

- Logic tested on **live data from multiple operator profiles**.
- Estimated potential annual savings: **4–7%** in roaming costs.
- Benefits across teams:
  - **Commercial teams:** Maximize financial outcomes, avoid commitment penalties.
  - **Engineering teams:** Ensure real-world feasibility with evidence-based recommendations.
  - **Finance teams:** Reduced billing disputes, faster cash flow, fewer manual errors.

**Key message:** This is not a theoretical saving — it was estimated on real operator data.

---

## Segment 8 — Technology Stack (10:00 – 10:45 | ~45 sec)

**Goal:** Build credibility with technical depth (keep it brief).

- **Infrastructure:** AWS S3 for scalability and availability.
- **Database layer:** Oracle SQL + in-memory tech (Redis, Hadoop) for diverse workloads.
- **Calculations:** Core processing in SQL; scoring via ML (K-means clustering).
- **Messaging:** ZeroMQ for real-time internal messaging.
- **Testing:** Component-level validation + full system verification with COT emulating core network functions.

**Key message:** Built for enterprise scale — speed, reliability, and flexibility.

---

## Segment 9 — Roadmap (10:45 – 11:30 | ~45 sec)

**Goal:** Show momentum and future vision.

**MACH – One Agreement roadmap:**
- Current: Steering recommendation logic live, leveraging group commitments and IoT effective rates.
- Short-term: IMSI-based agreements (currently in TAP flow).
- Next: Cross-commitment calculations.
- Following: BCE integration.

**TOMIA – IPN roadmap:**
- Current: Network quality scoring, region/city-level differentiated steering.
- Next upgrade: **AI agent** — chatbot-assisted traffic split distribution + troubleshooting and configuration recommendations.

**Key message:** This is a living product with a clear, ambitious roadmap.

---

## Segment 10 — Closing Statement (11:30 – 12:00 | ~30 sec)

**Goal:** Leave a strong, memorable final impression.

- One Steering aligns commercial strategy with network execution — turning insights into automated actions.
- It gives operators a **holistic perspective and enhanced control** of their business.
- Up to **7% annual cost savings**, stronger cross-team collaboration, and a platform built by **50+ combined years of roaming expertise**.
- Call to action: *"This is the future of wholesale roaming management — and we're already live."*

---

## Quick Reference: Key Numbers to Remember

| Metric | Value |
|---|---|
| Agreement multiplication factor | 4–5× per partner |
| Estimated annual cost savings | 4–7% |
| Combined industry experience | 50+ years (TOMIA + MACH) |
| Showcase event | Genesis Lion's Den, June 15–16, Limerick, Ireland |
| Contest | The Innovators 2026 by ROCCO |

---

## Topics Checklist for Manager Meeting

- [ ] Problem framing: commercial complexity (IoT growth, settlement mechanisms)
- [ ] Problem framing: network complexity (segment diversity, 2G/3G phase-out)
- [ ] Industry gap: silos, partial automation, lack of cross-service tools
- [ ] Solution overview: One Steering = One Agreement (MACH) + IPN (TOMIA)
- [ ] Real-world examples: group commitments, cross-commitments, city-level steering
- [ ] Business value: 4–7% savings, multi-team benefits
- [ ] Technology stack (brief)
- [ ] Roadmap: near-term priorities + AI agent vision
- [ ] Closing message: closed-loop, 50+ years of expertise, already live
- [ ] Contest context: The Innovators 2026, ROCCO Genesis June 15–16 in Limerick
