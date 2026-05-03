import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const outputDir = path.join(repoRoot, "test_assets", "agent_path_suite", "data_analyst");

async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (error) {
    const nodeModules = process.env.CODEX_NODE_MODULES;
    if (!nodeModules) {
      throw new Error(
        "Unable to load @oai/artifact-tool. Set CODEX_NODE_MODULES to the bundled node_modules path.",
        { cause: error },
      );
    }
    const modulePath = path.join(nodeModules, "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
    return import(pathToFileURL(modulePath).href);
  }
}

const orders = [
  {
    order_id: "O-1001",
    order_date: "2026-01-05",
    customer_id: "C-001",
    region_code: "NE",
    product_line: "Core",
    units: 10,
    unit_price_usd: 1200,
    discount_pct: 0.05,
    unit_cost_usd: 720,
    channel: "Web",
    status: "Complete",
    invoice_id: "INV-001",
    notes: "standard order",
  },
  {
    order_id: "O-1002",
    order_date: "01/07/2026",
    customer_id: "C-002",
    region_code: "SW",
    product_line: "Analytics",
    units: 5,
    unit_price_usd: 2500,
    discount_pct: 0.1,
    unit_cost_usd: 1600,
    channel: "Partner",
    status: "Complete",
    invoice_id: "INV-002",
    notes: "partial return later",
  },
  {
    order_id: "O-1003",
    order_date: "2026/01/09",
    customer_id: "C-003",
    region_code: "NE",
    product_line: "Automation",
    units: 7,
    unit_price_usd: 1800,
    discount_pct: 0,
    unit_cost_usd: 1100,
    channel: "Direct",
    status: "Complete",
    invoice_id: "INV-003",
    notes: "clean",
  },
  {
    order_id: "O-1004",
    order_date: "Jan 13 2026",
    customer_id: "C-004",
    region_code: "MW",
    product_line: "Core",
    units: 12,
    unit_price_usd: 1150,
    discount_pct: 0.12,
    unit_cost_usd: 700,
    channel: "Web",
    status: "Complete",
    invoice_id: "INV-004",
    notes: "return and restock fee",
  },
  {
    order_id: "O-1005",
    order_date: "2026-01-18",
    customer_id: "C-005",
    region_code: "SE",
    product_line: "Billing",
    units: 18,
    unit_price_usd: 450,
    discount_pct: 0,
    unit_cost_usd: 260,
    channel: "Retail",
    status: "Complete",
    invoice_id: "INV-005",
    notes: "low ASP",
  },
  {
    order_id: "O-1006",
    order_date: "2026-01-20",
    customer_id: "C-006",
    region_code: "SW",
    product_line: "Core",
    units: 9,
    unit_price_usd: 1250,
    discount_pct: 0.15,
    unit_cost_usd: 740,
    channel: "Direct",
    status: "Complete",
    invoice_id: "INV-006",
    notes: "large discount",
  },
  {
    order_id: "O-1007",
    order_date: "2026-02-02",
    customer_id: "C-007",
    region_code: "NE",
    product_line: "Analytics",
    units: 6,
    unit_price_usd: 2600,
    discount_pct: 0.08,
    unit_cost_usd: 1650,
    channel: "Partner",
    status: "Complete",
    invoice_id: "INV-007",
    notes: "partner sourced",
  },
  {
    order_id: "O-1008",
    order_date: "02/04/2026",
    customer_id: "C-008",
    region_code: "MW",
    product_line: "Automation",
    units: 4,
    unit_price_usd: 1900,
    discount_pct: 0,
    unit_cost_usd: 1150,
    channel: "Web",
    status: "Complete",
    invoice_id: "INV-008",
    notes: "",
  },
  {
    order_id: "O-1009",
    order_date: "2026-02-10",
    customer_id: "C-009",
    region_code: "SE",
    product_line: "Core",
    units: 15,
    unit_price_usd: 1180,
    discount_pct: 0.05,
    unit_cost_usd: 710,
    channel: "Direct",
    status: "Complete",
    invoice_id: "INV-009",
    notes: "standard discount",
  },
  {
    order_id: "O-1010",
    order_date: "2026-02-14",
    customer_id: "C-010",
    region_code: "SW",
    product_line: "Billing",
    units: 20,
    unit_price_usd: 430,
    discount_pct: 0.2,
    unit_cost_usd: 255,
    channel: "Retail",
    status: "Complete",
    invoice_id: "INV-010",
    notes: "duplicate invoice appears in CSV fixture",
  },
  {
    order_id: "O-1011",
    order_date: "2026-02-19",
    customer_id: "C-011",
    region_code: "NE",
    product_line: "Core",
    units: 8,
    unit_price_usd: 1210,
    discount_pct: 0,
    unit_cost_usd: 730,
    channel: "Web",
    status: "Complete",
    invoice_id: "INV-011",
    notes: "",
  },
  {
    order_id: "O-1012",
    order_date: "2026-02-22",
    customer_id: "C-012",
    region_code: "SE",
    product_line: "Analytics",
    units: 3,
    unit_price_usd: 2550,
    discount_pct: 0.05,
    unit_cost_usd: 1605,
    channel: "Partner",
    status: "Complete",
    invoice_id: "INV-012",
    notes: "small enterprise upsell",
  },
  {
    order_id: "O-1013",
    order_date: "2026-03-03",
    customer_id: "C-013",
    region_code: "MW",
    product_line: "Core",
    units: 11,
    unit_price_usd: 1190,
    discount_pct: 0.05,
    unit_cost_usd: 705,
    channel: "Direct",
    status: "Complete",
    invoice_id: "INV-013",
    notes: "",
  },
  {
    order_id: "O-1014",
    order_date: "03/08/2026",
    customer_id: "C-014",
    region_code: "SW",
    product_line: "Automation",
    units: 5,
    unit_price_usd: 1880,
    discount_pct: 0.1,
    unit_cost_usd: 1140,
    channel: "Web",
    status: "Complete",
    invoice_id: "INV-014",
    notes: "",
  },
  {
    order_id: "O-1015",
    order_date: "2026-03-12",
    customer_id: "C-015",
    region_code: "NE",
    product_line: "Billing",
    units: 25,
    unit_price_usd: 440,
    discount_pct: 0,
    unit_cost_usd: 265,
    channel: "Retail",
    status: "Complete",
    invoice_id: "INV-015",
    notes: "setup issue",
  },
  {
    order_id: "O-1016",
    order_date: "2026-03-19",
    customer_id: "C-016",
    region_code: "SE",
    product_line: "Automation",
    units: 6,
    unit_price_usd: 1840,
    discount_pct: 0.07,
    unit_cost_usd: 1125,
    channel: "Partner",
    status: "Complete",
    invoice_id: "INV-016",
    notes: "return and restock fee",
  },
];

const returns = [
  {
    return_id: "R-2001",
    order_id: "O-1002",
    return_date: "2026-01-28",
    units_returned: 1,
    reason: "customer_cancellation",
    restock_fee_usd: 0,
  },
  {
    return_id: "R-2002",
    order_id: "O-1004",
    return_date: "2026-02-02",
    units_returned: 2,
    reason: "damaged_in_transit",
    restock_fee_usd: 75,
  },
  {
    return_id: "R-2003",
    order_id: "O-1010",
    return_date: "2026-02-21",
    units_returned: 5,
    reason: "billing_error",
    restock_fee_usd: 0,
  },
  {
    return_id: "R-2004",
    order_id: "O-1015",
    return_date: "2026-03-27",
    units_returned: 3,
    reason: "incomplete_setup",
    restock_fee_usd: 50,
  },
  {
    return_id: "R-2005",
    order_id: "O-1016",
    return_date: "2026-03-29",
    units_returned: 1,
    reason: "damaged_in_transit",
    restock_fee_usd: 40,
  },
];

const regionTargets = [
  {
    region_code: "NE",
    region_name: "Northeast",
    q1_target_revenue_usd: 56000,
    target_margin_pct: 0.35,
    manager: "A. Rivera",
  },
  {
    region_code: "SW",
    region_name: "Southwest",
    q1_target_revenue_usd: 35000,
    target_margin_pct: 0.32,
    manager: "J. Patel",
  },
  {
    region_code: "MW",
    region_name: "Midwest",
    q1_target_revenue_usd: 32000,
    target_margin_pct: 0.34,
    manager: "N. Brooks",
  },
  {
    region_code: "SE",
    region_name: "Southeast",
    q1_target_revenue_usd: 41000,
    target_margin_pct: 0.35,
    manager: "R. Chen",
  },
];

const customerAccounts = [
  ["C-001", "Enterprise", "2024-03-15", "A. Rivera", "low"],
  ["C-002", "Mid-Market", "2025-01-03", "J. Patel", "medium"],
  ["C-003", "SMB", "2025-09-20", "A. Rivera", "low"],
  ["C-004", "Enterprise", "2023-11-08", "N. Brooks", "high"],
  ["C-005", "SMB", "2026-01-02", "R. Chen", "medium"],
  ["C-006", "Mid-Market", "2024-07-17", "J. Patel", "medium"],
  ["C-007", "Enterprise", "2022-05-25", "A. Rivera", "low"],
  ["C-008", "SMB", "2025-05-13", "N. Brooks", "low"],
  ["C-009", "Mid-Market", "2024-10-30", "R. Chen", "medium"],
  ["C-010", "SMB", "2025-12-04", "J. Patel", "high"],
  ["C-011", "Enterprise", "2023-02-14", "A. Rivera", "low"],
  ["C-012", "Mid-Market", "2025-08-22", "R. Chen", "low"],
  ["C-013", "SMB", "2025-06-01", "N. Brooks", "medium"],
  ["C-014", "Mid-Market", "2024-04-19", "J. Patel", "medium"],
  ["C-015", "Enterprise", "2022-12-09", "A. Rivera", "medium"],
  ["C-016", "SMB", "2025-10-11", "R. Chen", "high"],
].map(([customer_id, segment, signup_date, account_owner, risk_flag]) => ({
  customer_id,
  segment,
  signup_date,
  account_owner,
  risk_flag,
}));

const feedbackRows = [
  {
    feedback_id: "F-001",
    submitted_at: "2026-04-01 09:20",
    channel: "email",
    account_segment: "Enterprise",
    severity: "high",
    csat_score: 2,
    comment: "The invoice doubled after the discount vanished; support blamed the plan migration.",
  },
  {
    feedback_id: "F-002",
    submitted_at: "2026-04-01 10:14",
    channel: "app",
    account_segment: "Mid-Market",
    severity: "critical",
    csat_score: 1,
    comment: "Great, another midnight outage during our renewal week.",
  },
  {
    feedback_id: "F-003",
    submitted_at: "2026-04-02 08:02",
    channel: "web",
    account_segment: "SMB",
    severity: "medium",
    csat_score: 3,
    comment: "The dashboard is cleaner, but exporting filtered rows still takes ages.",
  },
  {
    feedback_id: "F-004",
    submitted_at: "2026-04-02 11:45",
    channel: "chat",
    account_segment: "Enterprise",
    severity: "low",
    csat_score: 5,
    comment: "Maria solved the SSO issue in 10 minutes. Super helpful.",
  },
  {
    feedback_id: "F-005",
    submitted_at: "2026-04-03 13:18",
    channel: "email",
    account_segment: "SMB",
    severity: "medium",
    csat_score: 2,
    comment: "Why does the basic plan cost more than last quarter with fewer included reports?",
  },
  {
    feedback_id: "F-006",
    submitted_at: "2026-04-04 15:33",
    channel: "app",
    account_segment: "Mid-Market",
    severity: "high",
    csat_score: 2,
    comment: "Sync says complete, yet half of the records are missing.",
  },
  {
    feedback_id: "F-007",
    submitted_at: "2026-04-05 09:03",
    channel: "web",
    account_segment: "Enterprise",
    severity: "low",
    csat_score: 4,
    comment: "The new keyboard shortcuts are fast once you find the settings page.",
  },
  {
    feedback_id: "F-008",
    submitted_at: "2026-04-06 16:21",
    channel: "chat",
    account_segment: "SMB",
    severity: "high",
    csat_score: 1,
    comment: "I waited 47 minutes and the agent pasted the same help article twice.",
  },
  {
    feedback_id: "F-009",
    submitted_at: "2026-04-07 12:40",
    channel: "email",
    account_segment: "Enterprise",
    severity: "medium",
    csat_score: 3,
    comment: "The renewal quote is probably fine, but I cannot reconcile the tax line.",
  },
  {
    feedback_id: "F-010",
    submitted_at: "2026-04-08 17:06",
    channel: "app",
    account_segment: "Mid-Market",
    severity: "critical",
    csat_score: 1,
    comment: "Export failed again after saying success. That is a fun definition of success.",
  },
  {
    feedback_id: "F-011",
    submitted_at: "2026-04-09 10:58",
    channel: "web",
    account_segment: "SMB",
    severity: "low",
    csat_score: 4,
    comment: "Gracias, the billing specialist fixed the tax exemption quickly.",
  },
  {
    feedback_id: "F-012",
    submitted_at: "2026-04-09 14:31",
    channel: "chat",
    account_segment: "Enterprise",
    severity: "medium",
    csat_score: 3,
    comment: "The page loads, then jumps around while I am editing a workflow.",
  },
  {
    feedback_id: "F-013",
    submitted_at: "2026-04-10 07:29",
    channel: "email",
    account_segment: "Mid-Market",
    severity: "high",
    csat_score: 2,
    comment: "API latency spiked every afternoon this week.",
  },
  {
    feedback_id: "F-014",
    submitted_at: "2026-04-10 18:15",
    channel: "app",
    account_segment: "SMB",
    severity: "medium",
    csat_score: 3,
    comment: "The product is okay, but the price jump is hard to justify.",
  },
  {
    feedback_id: "F-015",
    submitted_at: "2026-04-11 08:52",
    channel: "web",
    account_segment: "Enterprise",
    severity: "low",
    csat_score: 5,
    comment: "Onboarding was smooth and the support rep anticipated our questions.",
  },
  {
    feedback_id: "F-016",
    submitted_at: "2026-04-11 13:12",
    channel: "chat",
    account_segment: "Mid-Market",
    severity: "medium",
    csat_score: "",
    comment: "",
  },
  {
    feedback_id: "F-017",
    submitted_at: "2026-04-12 09:42",
    channel: "email",
    account_segment: "Enterprise",
    severity: "high",
    csat_score: 2,
    comment: "Two admins cannot log in after the role migration.",
  },
  {
    feedback_id: "F-018",
    submitted_at: "2026-04-12 16:48",
    channel: "app",
    account_segment: "SMB",
    severity: "medium",
    csat_score: 3,
    comment: "I like the reports, I do not like needing five clicks to schedule one.",
  },
];

function currency(value) {
  const formatted = Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return value < 0 ? `($${formatted})` : `$${formatted}`;
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function orderGross(order) {
  return order.units * order.unit_price_usd * (1 - order.discount_pct);
}

function returnForOrder(order) {
  const match = returns.find((item) => item.order_id === order.order_id);
  if (!match) {
    return { returnValue: 0, restockFee: 0, unitsReturned: 0 };
  }
  return {
    returnValue: match.units_returned * order.unit_price_usd * (1 - order.discount_pct),
    restockFee: match.restock_fee_usd,
    unitsReturned: match.units_returned,
  };
}

function orderNet(order) {
  const { returnValue, restockFee } = returnForOrder(order);
  return orderGross(order) - returnValue + restockFee;
}

function orderCostAfterReturns(order) {
  const { unitsReturned } = returnForOrder(order);
  return (order.units - unitsReturned) * order.unit_cost_usd;
}

function escapeCsvValue(value) {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filename, rows) {
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => escapeCsvValue(row[header])).join(",")),
  ];
  await fs.writeFile(path.join(outputDir, filename), `${lines.join("\n")}\n`, "utf8");
}

function worksheetRange(rowCount, colCount) {
  let col = "";
  let n = colCount;
  while (n > 0) {
    const rem = (n - 1) % 26;
    col = String.fromCharCode(65 + rem) + col;
    n = Math.floor((n - 1) / 26);
  }
  return `A1:${col}${rowCount}`;
}

function tableMatrix(rows) {
  const headers = Object.keys(rows[0]);
  return [headers, ...rows.map((row) => headers.map((header) => row[header]))];
}

function writeSheet(workbook, sheetName, rows) {
  const sheet = workbook.worksheets.add(sheetName);
  const matrix = tableMatrix(rows);
  sheet.getRange(worksheetRange(matrix.length, matrix[0].length)).values = matrix;
  return sheet;
}

function buildSalesCsvRows() {
  const cleanRows = orders.map((order, index) => {
    const gross = orderGross(order);
    const { returnValue } = returnForOrder(order);
    const net = orderNet(order);
    return {
      sale_id: `S-${String(index + 1).padStart(3, "0")}`,
      invoice_id: order.invoice_id,
      order_date: order.order_date,
      region_code: order.region_code,
      product_line: order.product_line,
      customer_segment: customerAccounts.find((account) => account.customer_id === order.customer_id).segment,
      units: index === 10 ? `${order.units} units` : order.units,
      gross_revenue_usd: currency(gross),
      discount_pct: pct(order.discount_pct),
      refund_usd: returnValue ? currency(returnValue) : "$0.00",
      net_revenue_usd: currency(net),
      marketing_spend_usd: index === 13 ? "" : currency(900 + index * 125),
      close_status: "Won",
      notes: order.notes,
    };
  });

  return [
    ...cleanRows,
    {
      ...cleanRows[9],
      sale_id: "S-017",
      notes: "intentional duplicate invoice row; should not be double-counted",
    },
    {
      sale_id: "S-018",
      invoice_id: "INV-018",
      order_date: "not set",
      region_code: "SW",
      product_line: "Analytics",
      customer_segment: "Mid-Market",
      units: "pending",
      gross_revenue_usd: "TBD",
      discount_pct: "15%",
      refund_usd: "$0.00",
      net_revenue_usd: "TBD",
      marketing_spend_usd: "$1,250.00",
      close_status: "Open",
      notes: "open pipeline row; exclude from won revenue summaries",
    },
  ];
}

function buildExpectedMetrics() {
  const byRegion = new Map();
  for (const order of orders) {
    const target = regionTargets.find((item) => item.region_code === order.region_code);
    const prior = byRegion.get(order.region_code) ?? {
      region_code: order.region_code,
      region_name: target.region_name,
      q1_target_revenue_usd: target.q1_target_revenue_usd,
      net_revenue_usd: 0,
      cost_after_returns_usd: 0,
      order_count: 0,
    };
    prior.net_revenue_usd += orderNet(order);
    prior.cost_after_returns_usd += orderCostAfterReturns(order);
    prior.order_count += 1;
    byRegion.set(order.region_code, prior);
  }

  const regionRows = [...byRegion.values()].map((row) => ({
    ...row,
    target_variance_usd: row.net_revenue_usd - row.q1_target_revenue_usd,
    margin_pct: (row.net_revenue_usd - row.cost_after_returns_usd) / row.net_revenue_usd,
  }));

  const bySegment = new Map();
  for (const order of orders) {
    const account = customerAccounts.find((item) => item.customer_id === order.customer_id);
    const prior = bySegment.get(account.segment) ?? {
      segment: account.segment,
      list_revenue_usd: 0,
      discount_leakage_usd: 0,
      order_count: 0,
    };
    const listValue = order.units * order.unit_price_usd;
    prior.list_revenue_usd += listValue;
    prior.discount_leakage_usd += listValue * order.discount_pct;
    prior.order_count += 1;
    bySegment.set(account.segment, prior);
  }

  const segmentRows = [...bySegment.values()].map((row) => ({
    ...row,
    discount_rate: row.discount_leakage_usd / row.list_revenue_usd,
  }));

  return {
    total_net_revenue_usd: regionRows.reduce((sum, row) => sum + row.net_revenue_usd, 0),
    missed_target_regions: regionRows
      .filter((row) => row.target_variance_usd < 0)
      .map((row) => row.region_code),
    by_region: regionRows,
    discount_leakage_by_segment: segmentRows,
  };
}

async function buildWorkbook(SpreadsheetFile, Workbook) {
  const workbook = Workbook.create();

  writeSheet(workbook, "orders", orders);
  writeSheet(workbook, "returns", returns);
  writeSheet(workbook, "region_targets", regionTargets);
  writeSheet(workbook, "customer_accounts", customerAccounts);
  writeSheet(workbook, "data_dictionary", [
    {
      field: "orders",
      definition: "Primary order grain. Revenue equals units * unit_price_usd * (1 - discount_pct).",
    },
    {
      field: "returns",
      definition: "Return records by order_id. Net revenue subtracts returned units and adds retained restock fees.",
    },
    {
      field: "region_targets",
      definition: "Q1 revenue and margin goals by region_code.",
    },
    {
      field: "customer_accounts",
      definition: "Customer segment, owner, and risk metadata for order joins.",
    },
    {
      field: "known_checks",
      definition: "Total net revenue should be 160953.50 after returns and restock fees.",
    },
  ]);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  for (const sheetName of ["orders", "returns", "region_targets", "customer_accounts", "data_dictionary"]) {
    const blob = await workbook.render({ sheetName, range: "A1:H12", scale: 1 });
    console.log(`rendered ${sheetName}: ${blob.size} bytes`);
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(outputDir, "ops_revenue_challenge.xlsx"));
}

async function main() {
  const { SpreadsheetFile, Workbook } = await loadArtifactTool();
  await fs.mkdir(outputDir, { recursive: true });

  await writeCsv("regional_sales_messy.csv", buildSalesCsvRows());
  await writeCsv("customer_feedback_edge_cases.csv", feedbackRows);
  await fs.writeFile(
    path.join(outputDir, "expected_metrics.json"),
    `${JSON.stringify(buildExpectedMetrics(), null, 2)}\n`,
    "utf8",
  );
  await buildWorkbook(SpreadsheetFile, Workbook);

  console.log(`Wrote analyst challenge assets to ${outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
