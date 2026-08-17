/**
 * The ballot — this console's signature element.
 *
 * The system does not produce a score, it produces a vote: four detectors that
 * genuinely disagree with each other, and a quorum. Rendering a single number
 * would hide exactly the mechanism this demonstration exists to show, so the
 * verdict is drawn as cast ballots with the quorum line in place. Reading
 * "three filled, line after two" tells you the rule and the result at once.
 */
export default function Ballot({ ensemble, size = "sm" }) {
  const { votes_for: votesFor, votes_total: total, votes_required: required, is_anomaly: isAnomaly } =
    ensemble;

  const cells = [];
  for (let i = 0; i < total; i += 1) {
    // The quorum line sits after the last cell that is still short of the
    // requirement, so its position encodes `votes_required` directly.
    if (i === required) {
      cells.push(<span className="quorum" key={`q${i}`} aria-hidden="true" />);
    }
    cells.push(
      <span className="cell" key={i} data-cast={i < votesFor} aria-hidden="true" />,
    );
  }

  return (
    <span
      className="ballot"
      data-size={size}
      data-anomaly={isAnomaly}
      role="img"
      aria-label={`${votesFor} of ${total} models flagged this; ${required} required`}
    >
      {cells}
    </span>
  );
}
