import { Input, Select } from "../ui/Input";
import type { EnvironmentIn, VesselIn } from "../../services/api";

/**
 * The model was trained on these exact features, so both blocks are shared by
 * the Prediction, Optimization and Scenario pages. Nothing is prefilled.
 */

export const EMPTY_VESSEL: VesselIn = { vessel_type: "", displacement: 0, trim: 0 };

export const EMPTY_ENVIRONMENT: EnvironmentIn = {
  wind_speed: 0,
  wind_direction_relative: 0,
  combined_wave_height: 0,
  combined_wave_period: 0,
  sea_current_speed: 0,
  sea_current_direction_relative: 0,
  sea_water_temperature: 0,
};

/** Typical values observed in the training data, shown as placeholders only. */
const HINTS = {
  displacement: "e.g. 12",
  trim: "e.g. 0",
  wind_speed: "e.g. 14",
  wind_direction_relative: "e.g. 90",
  combined_wave_height: "e.g. 3",
  combined_wave_period: "e.g. 5.5",
  sea_current_speed: "e.g. 0.5",
  sea_current_direction_relative: "e.g. 90",
  sea_water_temperature: "e.g. 17",
};

export function vesselIsComplete(v: VesselIn): boolean {
  return Boolean(v.vessel_type) && v.displacement > 0;
}

export function environmentIsComplete(e: EnvironmentIn): boolean {
  return e.combined_wave_period > 0;
}

export function VesselFields({
  value,
  onChange,
  vesselTypes,
  disabled = false,
}: {
  value: VesselIn;
  onChange: (next: VesselIn) => void;
  vesselTypes: string[];
  disabled?: boolean;
}) {
  const set = <K extends keyof VesselIn>(k: K, v: VesselIn[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <Select
        label="Vessel type"
        value={value.vessel_type}
        disabled={disabled || vesselTypes.length === 0}
        onChange={(e) => set("vessel_type", e.target.value)}
        aria-label="Vessel type"
      >
        <option value="">{vesselTypes.length ? "Select…" : "Loading from model…"}</option>
        {vesselTypes.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </Select>
      <Input
        label="Displacement"
        type="number"
        min={0}
        step={0.1}
        suffix="kt"
        placeholder={HINTS.displacement}
        value={value.displacement || ""}
        onChange={(e) => set("displacement", Number(e.target.value))}
      />
      <Input
        label="Trim"
        type="number"
        step={0.1}
        suffix="m"
        placeholder={HINTS.trim}
        value={value.trim || ""}
        onChange={(e) => set("trim", Number(e.target.value))}
      />
    </div>
  );
}

export function EnvironmentFields({
  value,
  onChange,
}: {
  value: EnvironmentIn;
  onChange: (next: EnvironmentIn) => void;
}) {
  const set = <K extends keyof EnvironmentIn>(k: K, v: EnvironmentIn[K]) => onChange({ ...value, [k]: v });
  const field = (
    key: keyof EnvironmentIn,
    label: string,
    suffix: string,
    step = 0.1
  ) => (
    <Input
      key={key}
      label={label}
      type="number"
      min={0}
      step={step}
      suffix={suffix}
      placeholder={HINTS[key]}
      value={value[key] || ""}
      onChange={(e) => set(key, Number(e.target.value))}
    />
  );

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {field("wind_speed", "Wind speed", "m/s")}
      {field("wind_direction_relative", "Wind dir. (rel.)", "°", 1)}
      {field("combined_wave_height", "Wave height", "m")}
      {field("combined_wave_period", "Wave period", "s")}
      {field("sea_current_speed", "Current speed", "kn", 0.01)}
      {field("sea_current_direction_relative", "Current dir. (rel.)", "°", 1)}
      {field("sea_water_temperature", "Sea temperature", "°C")}
    </div>
  );
}
