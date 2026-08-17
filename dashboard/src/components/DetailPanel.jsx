import Ballot from "./Ballot.jsx";
import DecisionBar from "./DecisionBar.jsx";
import ShapChart from "./ShapChart.jsx";

const DETECTOR_LABELS = {
  isolation_forest: "Isolation Forest",
  lof: "Local Outlier Factor",
  one_class_svm: "One-Class SVM",
  dbscan: "DBSCAN",
};

export default function DetailPanel({ result, decision, onDecide }) {
  const { ensemble, detectors, explanation, warnings } = result;

  return (
    <div className="detail">
      <div className="verdict-block">
        <div>
          <div className="micro" style={{ marginBottom: 8 }}>
            Ensemble verdict
          </div>
          <Ballot ensemble={ensemble} size="lg" />
          <div className="ballot-legend">
            <span className="verdict" data-anomaly={ensemble.is_anomaly}>
              {ensemble.is_anomaly ? "Flagged" : "Clear"}
            </span>
            <span style={{ fontSize: 12, color: "var(--ink-dim)" }}>
              <span className="num">{ensemble.votes_for}</span> of{" "}
              <span className="num">{ensemble.votes_total}</span> models agree ·{" "}
              <span className="num">{ensemble.votes_required}</span> needed
            </span>
          </div>
        </div>

        <div>
          <div className="micro" style={{ marginBottom: 8 }}>
            How each model voted
          </div>
          <div className="detectors">
            {detectors.map((d) => (
              <div className="det" key={d.name} data-cast={d.flag === 1}>
                <span className="mark" aria-hidden="true" />
                <span className="dname">{DETECTOR_LABELS[d.name] ?? d.name}</span>
                <span className="pct num" title="Percentile against this model's training scores">
                  {d.score_percentile.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
          <p className="footnote">
            Percentiles rank against each model's own training scores. Four
            other models — MCD, GMM, K-Means, PCA-reconstruction — are fitted
            during training but do not vote.
          </p>
        </div>

        <DecisionBar result={result} decision={decision} onDecide={onDecide} />
      </div>

      <div style={{ display: "grid", gap: 18, alignContent: "start" }}>
        <ShapChart features={explanation.top_features} />

        <div>
          <div className="micro" style={{ marginBottom: 8 }}>
            Explanation
          </div>
          <p className="sentence" style={{ margin: 0 }}>
            {explanation.plain_english}
          </p>
        </div>

        {warnings?.length > 0 && (
          <div>
            <div className="micro" style={{ marginBottom: 6 }}>
              Data notes
            </div>
            <div className="warnings">
              {warnings.map((w) => (
                <span className="warn" key={w}>
                  {w}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
