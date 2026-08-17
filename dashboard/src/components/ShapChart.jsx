const FEATURE_LABELS = {
  TransactionAmount: "Transaction amount",
  CustomerAge: "Customer age",
  TransactionDuration: "Transaction duration",
  LoginAttempts: "Login attempts",
  AccountBalance: "Account balance",
  TimeSinceLastTx_Hours: "Gap since last transaction",
  DailyAccountVolume: "Transactions on account today",
  UtilizationRatio: "Share of balance drained",
  DailyDeviceVelocity: "Transactions from device today",
  Location_Freq: "Location familiarity",
  TransactionType_Credit: "Credit transaction",
  TransactionType_Debit: "Debit transaction",
  Channel_ATM: "ATM",
  Channel_Branch: "In-branch",
  Channel_Online: "Online",
  CustomerOccupation_Doctor: "Occupation: doctor",
  CustomerOccupation_Engineer: "Occupation: engineer",
  CustomerOccupation_Retired: "Occupation: retired",
  CustomerOccupation_Student: "Occupation: student",
};

const compact = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

// The nine one-hot columns are indicators, not quantities: printing "0" beside
// "Credit transaction" makes the reader decode it. Show yes/no instead.
const ONE_HOT = new Set([
  "TransactionType_Credit",
  "TransactionType_Debit",
  "Channel_ATM",
  "Channel_Branch",
  "Channel_Online",
  "CustomerOccupation_Doctor",
  "CustomerOccupation_Engineer",
  "CustomerOccupation_Retired",
  "CustomerOccupation_Student",
]);

function readableValue(feature, value) {
  if (ONE_HOT.has(feature)) return value >= 0.5 ? "yes" : "no";
  // Match the thousands separators used everywhere else, so a feature value
  // reads the same way as the amount in the row above it.
  return compact.format(value);
}

/**
 * Signed contributions, drawn from a centre axis.
 *
 * The sign is the whole point — SHAP says which way each feature pushed — so a
 * one-sided bar chart would throw away half the information. Bars are scaled
 * against the largest absolute contribution in this explanation, so the widest
 * bar always fills its half and the comparison stays legible whatever the
 * absolute magnitudes happen to be.
 */
export default function ShapChart({ features }) {
  if (!features?.length) return null;

  const widest = Math.max(...features.map((f) => Math.abs(f.shap_value))) || 1;

  return (
    <div>
      <div className="micro" style={{ marginBottom: 8 }}>
        Feature contributions
      </div>

      <div className="shap">
        {features.map((f) => {
          const width = (Math.abs(f.shap_value) / widest) * 50;
          const label = FEATURE_LABELS[f.feature] ?? f.feature;
          return (
            <div className="shap-row" key={f.feature}>
              <div className="shap-label" title={`${label} = ${readableValue(f.feature, f.value)}`}>
                {label}{" "}
                <span className="num" style={{ color: "var(--ink-faint)" }}>
                  {readableValue(f.feature, f.value)}
                </span>
              </div>
              <div className="shap-track">
                <span className="axis" aria-hidden="true" />
                <span
                  className="shap-bar"
                  data-dir={f.direction}
                  style={{ width: `${width}%` }}
                  role="img"
                  aria-label={`${label} ${
                    f.direction === "increases" ? "pushes toward" : "pushes away from"
                  } anomaly`}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="shap-scale">
        <span>← toward normal</span>
        <span>toward anomaly →</span>
      </div>
    </div>
  );
}
