import React, { useEffect, useRef } from "react";
import HeroCanvas from "./components/HeroCanvas.jsx";
import Studio from "./components/Studio.jsx";

function useReveal() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll(".reveal"));
    if (!("IntersectionObserver" in window)) {
      els.forEach((e) => e.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) {
            en.target.classList.add("in");
            io.unobserve(en.target);
          }
        });
      },
      { threshold: 0.16 }
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function Nav() {
  return (
    <nav className="nav">
      <a className="nav__brand" href="#top">
        RHINOFORM<span>°</span>
      </a>
      <div className="nav__links">
        <a href="#thesis">Approach</a>
        <a href="#studio">Studio</a>
        <a href="#method">Pipeline</a>
        <a href="#responsible">Responsible use</a>
      </div>
      <span className="nav__tag">Research prototype</span>
    </nav>
  );
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="hero__bg">
        <img className="hero__poster" src="/hero_poster.png" alt="" aria-hidden="true" />
        <HeroCanvas />
        <div className="hero__mask" />
      </div>
      <div className="hero__inner">
        <span className="hero__eyebrow">Population-informed sparse-control nasal preview</span>
        <h1 className="hero__title">
          An instant anchor,
          <br />
          and a certified enhancement.
        </h1>
        <p className="hero__lede">
          Rhinoform pairs a frozen linear Ridge anchor — instant while you drag — with a learned
          Certified&nbsp;RB-SR enhancement that is only shown after it passes an anchor-relative
          fold-subset certificate. When the certificate, the runtime or the timeout fails, the
          preview automatically stays on the anchor.
        </p>
        <div className="hero__cta">
          <a className="btn" href="#studio">
            Open the studio
          </a>
          <a className="btn btn--ghost" href="#method">
            See the pipeline
          </a>
        </div>
      </div>
      <div className="hero__scroll" aria-hidden="true">
        scroll
      </div>
    </header>
  );
}

function Disclaimer() {
  return (
    <div className="disclaimer" role="note">
      <strong>Research prototype — not a medical device.</strong> Outputs do not predict surgical or
      anatomical results. Coordinates are uncalibrated FaceScape model units. For morphology
      communication only.
    </div>
  );
}

function Section({ id, index, kicker, title, children }) {
  return (
    <section className="section reveal" id={id}>
      <div className="section__head">
        <span className="section__index">{index}</span>
        <span className="section__kicker">{kicker}</span>
      </div>
      <h2 className="section__title">{title}</h2>
      <div className="section__body">{children}</div>
    </section>
  );
}

const PRINCIPLES = [
  {
    h: "Anchor first",
    p: "A frozen PCA-64 Ridge operator answers every drag instantly. It is deterministic, auditable and the fallback for everything else.",
  },
  {
    h: "Enhancement must earn its place",
    p: "The learned residual (CVAE proposal shaped by a spatial gate) is only shown after an iterative projection certifies that its fold set is a subset of the anchor's.",
  },
  {
    h: "Two separate guards",
    p: "An input-applicability policy limits requests outside the evaluated control range; the residual certificate governs the enhancement. They are never blended into one score.",
  },
  {
    h: "Fail closed",
    p: "Model hashes are locked. A mismatch, a timeout or a certificate failure never interrupts the session — the preview simply stays on the Ridge anchor and says so.",
  },
];

const METHOD = [
  ["01", "Population template", "The demo edits the training-identity mean nasal region, shown in a training-mean full-head context — population statistics. No individual scan is shipped to the browser."],
  ["02", "Sparse controls", "Six semantic sliders write sparse, role-restricted displacements to nine nasal landmarks (a 27-D control vector); roles come from the frozen nasal-subunit partition. The displayed deformation is the learned field itself — no display-frame handle enforcement."],
  ["03", "Applicability check", "The request is compared against the frozen validation control distribution; edits beyond the evaluated range are scaled back or withheld."],
  ["04", "Instant Ridge anchor", "The exact frozen ridge weights (PCA-64 source basis, λ = 300) run in the browser and update the mesh on every slider event — the smooth population deformation, which realises only about a third of the requested handle motion."],
  ["05", "Certified RB-SR", "Asynchronously — in the browser itself — the frozen CVAE proposes a dense residual, the learned gate shapes it, and an iterative projection (attenuation 0.75, ≤ 64 rounds, 1001-step uniform fallback) certifies a fold subset relative to the anchor. The certified correction roughly doubles how much of the requested edit is realised."],
  ["06", "Commit or fall back", "A certified result replaces the anchor smoothly; any failure keeps the anchor. The session log records requested and effective controls, certificate status and model hashes."],
];

const INSTRUMENTS = [
  ["Fold-subset certificate", "The deployed guarantee: the certified result introduces no orientation-reversed faces beyond those already present in the Ridge anchor."],
  ["Residual retention", "How much of the learned proposal survived projection — shown, not hidden, whenever attenuation was required."],
  ["Edge-strain p95", "Relative edge-length distortion highlights over-stretched regions in the risk overlay."],
  ["Evaluated-range distance", "A χ statistic in the frozen validation control frame keeps requests inside the range the system was actually evaluated on."],
];

export default function App() {
  useReveal();
  const year = useRef(new Date().getFullYear()).current;
  return (
    <div className="page">
      <Nav />
      <Hero />
      <Disclaimer />

      <main>
        <div className="section-wrap">
        <Section
          id="thesis"
          index="—"
          kicker="The approach"
          title="For consultation preview, pair the fastest trustworthy model with a certified enhancement."
        >
          <p>
            The evaluation behind this demo found a consistent accuracy–regularity trade-off: more
            expressive editors gain accuracy but introduce geometric distortions that a consultation
            preview cannot tolerate silently. Rhinoform's answer is architectural — an instant linear
            anchor for interaction, a learned enhancement admitted only under an explicit geometric
            certificate, and an automatic, visible fallback.
          </p>
          <div className="cards">
            {PRINCIPLES.map((c) => (
              <article className="card reveal" key={c.h}>
                <h3>{c.h}</h3>
                <p>{c.p}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section
          id="studio"
          index="01"
          kicker="Interactive studio"
          title="Drag a control. The anchor answers instantly; the enhancement follows when certified."
        >
          <p className="section__note">
            Consultation mode speaks plainly; research mode exposes the certificate, retention,
            projection rounds, fold counts and model hashes. Every edit is certified live by the
            frozen pipeline running in the browser, verified against the authoritative Python
            runtime on golden cases; a local research service can serve the same chain for
            licensed identity sources.
          </p>
        </Section>
        </div>

        <div className="studio-frame reveal">
          <Studio />
        </div>

        <div className="section-wrap">
        <Section id="method" index="02" kicker="Pipeline" title="The deployed chain is the paper's frozen final pipeline.">
          <p className="section__note">
            Every stage below is bound by SHA-256 to the frozen final-rerun artifacts (base model,
            spatial gate and projection policy). If any identity check fails, the certified stage is
            disabled and the demo serves the Ridge anchor only.
          </p>
          <ol className="steps">
            {METHOD.map(([n, h, p]) => (
              <li className="step reveal" key={n}>
                <span className="step__n">{n}</span>
                <div>
                  <h4>{h}</h4>
                  <p>{p}</p>
                </div>
              </li>
            ))}
          </ol>
        </Section>

        <Section id="instrumentation" index="03" kicker="Instrumentation" title="The paper's diagnostics, visible in the interface.">
          <div className="cards">
            {INSTRUMENTS.map(([h, p]) => (
              <article className="card reveal" key={h}>
                <h3>{h}</h3>
                <p>{p}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section
          id="responsible"
          index="04"
          kicker="Responsible innovation"
          title="An honest demo, not a hero demo."
        >
          <p>
            This is a morphology-communication tool that deliberately avoids over-claiming. The
            certificate is a geometric statement relative to the Ridge anchor — it is never described
            as clinical safety, and no output predicts a surgical result. The public build ships only
            population statistics derived from the licensed dataset; individual scans are never
            redistributed. A user-supplied OBJ mesh (registered to the frozen correspondence) is
            parsed and bound entirely inside the browser and never transmitted to any server.
          </p>
          <p className="section__note">
            Mandatory framing: research prototype · not a medical device · outputs do not predict
            surgical or anatomical results · uncalibrated model units.
          </p>
        </Section>
        </div>
      </main>

      <footer className="foot">
        <div className="foot__brand">RHINOFORM°</div>
        <p>
          Population-informed sparse-control nasal preview · frozen Ridge anchor + Certified RB-SR ·{" "}
          {year}. Research prototype. Not a medical device.
        </p>
      </footer>
    </div>
  );
}
