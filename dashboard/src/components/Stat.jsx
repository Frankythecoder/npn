/**
 * One counted figure from a set of scored transactions.
 *
 * `tone` colours the value: high for flagged, away for clear, dim for a count
 * that is only a footnote to the others.
 */
export default function Stat({ value, label, tone }) {
  return (
    <div className="batch-stat" data-tone={tone}>
      <span className="num v">{value.toLocaleString()}</span>
      <span className="micro">{label}</span>
    </div>
  );
}
