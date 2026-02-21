#!/usr/bin/env bash
set -euo pipefail

MILESTONE="${1:-v17c-april-2026}"
REPO="${2:-ealtenau/SWORD}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS_DOC="${ROOT_DIR}/docs/roadmaps/v17c_status.md"
PLAN_DOC="${ROOT_DIR}/docs/roadmaps/v17c_plan_to_april.md"
ROADMAP_DOC="${ROOT_DIR}/docs/roadmaps/v17c_v18_roadmap.md"

for cmd in gh jq rg sed; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 2
  fi
done

milestones_json="$(gh api "repos/${REPO}/milestones?state=all&per_page=100")"
due_date="$(
  echo "${milestones_json}" | jq -r --arg m "${MILESTONE}" '
    [.[] | select(.title == $m) | .due_on][0] // empty
  ' | sed -E 's/T.*$//'
)"

if [[ -z "${due_date}" ]]; then
  echo "Milestone '${MILESTONE}' not found in ${REPO}" >&2
  exit 2
fi

live_open="$(gh issue list --state open --limit 500 --search "milestone:\"${MILESTONE}\"" --json number | jq 'length')"
live_closed="$(gh issue list --state closed --limit 500 --search "milestone:\"${MILESTONE}\"" --json number | jq 'length')"
live_v18_open="$(gh issue list --state open --limit 500 --search "label:v18-deferred" --json number | jq 'length')"

status_open="$(rg '^\| Open \| [0-9]+ \|$' "${STATUS_DOC}" -o -r '$0' | head -n1 | sed -E 's/^\| Open \| ([0-9]+) \|$/\1/')"
status_closed="$(rg '^\| Closed \| [0-9]+ \|$' "${STATUS_DOC}" -o -r '$0' | head -n1 | sed -E 's/^\| Closed \| ([0-9]+) \|$/\1/')"
status_due="$(rg '^\| Due date \| [0-9]{4}-[0-9]{2}-[0-9]{2} \|$' "${STATUS_DOC}" -o -r '$0' | head -n1 | sed -E 's/^\| Due date \| ([0-9]{4}-[0-9]{2}-[0-9]{2}) \|$/\1/')"

plan_open="$(rg '^- Open: [0-9]+$' "${PLAN_DOC}" -o -r '$0' | head -n1 | sed -E 's/^- Open: ([0-9]+)$/\1/')"
plan_closed="$(rg '^- Closed: [0-9]+$' "${PLAN_DOC}" -o -r '$0' | head -n1 | sed -E 's/^- Closed: ([0-9]+)$/\1/')"
plan_due="$(rg '^- Due date: [0-9]{4}-[0-9]{2}-[0-9]{2}$' "${PLAN_DOC}" -o -r '$0' | head -n1 | sed -E 's/^- Due date: ([0-9]{4}-[0-9]{2}-[0-9]{2})$/\1/')"

roadmap_v17_line="$(rg '^\| \*\*v17c\*\* .* \| [0-9]+ open / [0-9]+ closed .* \|$' "${ROADMAP_DOC}" -o -r '$0' | head -n1)"
roadmap_v17_open="$(echo "${roadmap_v17_line}" | sed -E 's/^.*\| ([0-9]+) open \/ ([0-9]+) closed.*$/\1/')"
roadmap_v17_closed="$(echo "${roadmap_v17_line}" | sed -E 's/^.*\| ([0-9]+) open \/ ([0-9]+) closed.*$/\2/')"

roadmap_v17_milestone_line="$(rg '^\| v17c-april-2026 \| Open \| [0-9]+ \| [0-9]+ \|.*$' "${ROADMAP_DOC}" -o -r '$0' | head -n1)"
roadmap_v17_milestone_open="$(echo "${roadmap_v17_milestone_line}" | sed -E 's/^\| v17c-april-2026 \| Open \| ([0-9]+) \| ([0-9]+) \|.*$/\1/')"
roadmap_v17_milestone_closed="$(echo "${roadmap_v17_milestone_line}" | sed -E 's/^\| v17c-april-2026 \| Open \| ([0-9]+) \| ([0-9]+) \|.*$/\2/')"

roadmap_v18_line="$(rg '^\| \*\*v18\*\* .* \| [0-9]+ open .* \|$' "${ROADMAP_DOC}" -o -r '$0' | head -n1)"
roadmap_v18_open="$(echo "${roadmap_v18_line}" | sed -E 's/^.*\| ([0-9]+) open.*$/\1/')"

echo "Live milestone '${MILESTONE}' in ${REPO}: open=${live_open}, closed=${live_closed}, due=${due_date}"
echo "Live v18 deferred open issues: ${live_v18_open}"

mismatch=0
check_val() {
  local label="$1"
  local doc_val="$2"
  local live_val="$3"
  if [[ "${doc_val}" != "${live_val}" ]]; then
    echo "MISMATCH ${label}: doc=${doc_val}, live=${live_val}"
    mismatch=1
  fi
}

check_val "v17c_status open" "${status_open}" "${live_open}"
check_val "v17c_status closed" "${status_closed}" "${live_closed}"
check_val "v17c_status due" "${status_due}" "${due_date}"

check_val "v17c_plan_to_april open" "${plan_open}" "${live_open}"
check_val "v17c_plan_to_april closed" "${plan_closed}" "${live_closed}"
check_val "v17c_plan_to_april due" "${plan_due}" "${due_date}"

check_val "v17c_v18_roadmap overview v17c open" "${roadmap_v17_open}" "${live_open}"
check_val "v17c_v18_roadmap overview v17c closed" "${roadmap_v17_closed}" "${live_closed}"
check_val "v17c_v18_roadmap milestone row open" "${roadmap_v17_milestone_open}" "${live_open}"
check_val "v17c_v18_roadmap milestone row closed" "${roadmap_v17_milestone_closed}" "${live_closed}"
check_val "v17c_v18_roadmap overview v18 open" "${roadmap_v18_open}" "${live_v18_open}"

if [[ "${mismatch}" -eq 1 ]]; then
  echo "Roadmap docs are OUT OF SYNC."
  exit 1
fi

echo "Roadmap docs are in sync."
