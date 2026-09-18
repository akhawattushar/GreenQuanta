import { type ReactNode } from "react";

export interface Column<T> {
  header: string;
  accessor: (row: T) => ReactNode;
  align?: "left" | "right" | "center";
}

const alignClass = (a?: string) => (a === "right" ? "text-right" : a === "center" ? "text-center" : "text-left");

export function Table<T>({
  columns,
  data,
  keyField,
}: {
  columns: Column<T>[];
  data: T[];
  keyField: (row: T) => string;
}) {
  return (
    <div className="w-full max-w-full overflow-x-auto rounded-xl border border-slate-200">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead className="bg-slate-50">
          <tr>
            {columns.map((c) => (
              <th
                key={c.header}
                scope="col"
                className={`whitespace-nowrap border-b border-slate-200 px-4 py-2.5 text-[11px] font-bold uppercase tracking-wide text-slate-500 ${alignClass(
                  c.align
                )}`}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={keyField(row)} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/70">
              {columns.map((c) => (
                <td key={c.header} className={`px-4 py-3 text-navy-800 ${alignClass(c.align)}`}>
                  {c.accessor(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
