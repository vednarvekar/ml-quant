// Script to auto-call Upstox API to download data of OHLCV

import axios from "axios";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

dotenv.config();

const SYMBOLS = {
  // INDEXES
  nifty50: "NSE_INDEX|Nifty 50",
  bankNifty: "NSE_INDEX|Nifty Bank",
  finnifty: "NSE_INDEX|Nifty Fin Service",

  // STABLE LARGE CAPS
  reliance: "NSE_EQ|INE002A01018",
  hdfcBank: "NSE_EQ|INE040A01034",
  tcs: "NSE_EQ|INE467B01029",
  infosys: "NSE_EQ|INE009A01021",
  hul: "NSE_EQ|INE030A01027",
  asianPaints: "NSE_EQ|INE021A01026",

  // MOMENTUM / TRENDING
  tataMotors: "NSE_EQ|INE155A01022",
  hal: "NSE_EQ|INE066F01020",
  bel: "NSE_EQ|INE263A01024",
  adaniPorts: "NSE_EQ|INE742F01042",
  jioFinancial: "NSE_EQ|INE758E01017",

  // VOLATILE / HIGH BETA
  // suzlon: "NSE_EQ|INE040H01021",
  // rvnl: "NSE_EQ|INE415G01027",
  // ireda: "NSE_EQ|INE202E01016",
  // yesBank: "NSE_EQ|INE528G01035",
  // vodafoneIdea: "NSE_EQ|INE669E01016",
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const BASE_DIR = path.resolve(__dirname, "..");

async function downloadOHLCVData() {
  const accessToken = process.env.upstoxToken;
  
  // Iterate through every symbol defined in the SYMBOLS object
  for (const [symbolName, instrumentValue] of Object.entries(SYMBOLS)) {
    // const symbolName = "jioFinancial";
    const instrumentKey = encodeURIComponent(instrumentValue);
    // const instrumentKey = SYMBOLS.jioFinancial;
    const data_path = path.join(BASE_DIR, "data", "raw", "1H", `${symbolName}_ohlcv.json`);

    // Ensure the directory exists
    const dir = path.dirname(data_path);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    // Start at the beginning of the year
    let currentYear = 2022; // Upstox data starts from 2006
    let currentMonth = 0; // January

    const now = new Date();
    const fetchedData: Array<{
      timestamp: string;
      open: number;
      high: number;
      low: number;
      close: number;
      volume: number;
      oi: number;
    }> = [];

    console.log(`--- Starting Extraction for: ${symbolName} ---`);

    while (new Date(currentYear, currentMonth, 1) < now) {
      const startOfMonth = new Date(currentYear, currentMonth, 1);
      const endOfMonth = new Date(currentYear, currentMonth + 1, 0);

      const fromDateStr = startOfMonth.toISOString().split("T")[0];
      const toDate = endOfMonth > now ? now : endOfMonth;
      const toDateStr = toDate.toISOString().split("T")[0];

      const url = `https://api.upstox.com/v3/historical-candle/${instrumentKey}/hours/1/${toDateStr}/${fromDateStr}`;

      try {
        await new Promise((resolve) => setTimeout(resolve, 100));
        console.log(`Fetching ${symbolName} [${fromDateStr} to ${toDateStr}]`);
        const response = await axios.get(url, {
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
        });

        if (response.data.data && response.data.data.candles) {
          const rawCandles = response.data.data.candles.reverse();
          const mappedCandles = rawCandles.map((c: [string, number, number, number, number, number, number]) => ({
            timestamp: c[0],
            open: c[1],
            high: c[2],
            low: c[3],
            close: c[4],
            volume: c[5],
            oi: c[6],
          }));

          fetchedData.push(...mappedCandles);
        }

        currentMonth++;
        if (currentMonth > 11) {
          currentMonth = 0;
          currentYear++;
        }
      } catch (error: any) {
        const errorData = error.response?.data;
        console.error(`Request Failed for ${symbolName} at ${fromDateStr}:`, JSON.stringify(errorData, null, 2));

        if (error.response?.status === 429) {
          console.log("Rate limited! Waiting 35s...");
          await new Promise((resolve) => setTimeout(resolve, 35000));
          continue;
        }
        break; 
      }
    }

    if (fetchedData.length > 0) {
      fs.writeFileSync(data_path, JSON.stringify(fetchedData, null, 2));
      console.log(`--- DONE: ${symbolName} saved (${fetchedData.length} candles) ---`);
    }
  }
  console.log("--- ALL SYMBOLS PROCESSED ---");
}

downloadOHLCVData();