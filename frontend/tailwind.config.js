/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          50: "#F2F6FB",
          100: "#E3EBF5",
          200: "#C6D6E9",
          300: "#9CB4D3",
          400: "#6484AE",
          500: "#3F5F8A",
          600: "#2B4770",
          700: "#1E355A",
          800: "#152747",
          900: "#0E1B33",
          950: "#081120",
        },
        teal: {
          50: "#EAF8F8",
          100: "#CFEFEF",
          200: "#9FDEDF",
          300: "#63C7C9",
          400: "#2FA9AC",
          500: "#0E8C90",
          600: "#0B7175",
          700: "#0A5B5F",
          800: "#09484B",
          900: "#073B3E",
        },
        brand: {
          green: "#16A34A",
          greenLight: "#E9F7EF",
          teal: "#0E8C90",
          tealLight: "#E7F5F6",
          navy: "#0E1B33",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px 0 rgba(16,24,40,0.04), 0 1px 3px 0 rgba(16,24,40,0.06)",
        cardHover: "0 4px 12px -2px rgba(16,24,40,0.10), 0 2px 6px -2px rgba(16,24,40,0.06)",
        panel: "0 8px 28px -8px rgba(16,24,40,0.18)",
      },
      borderRadius: { xl: "0.75rem", "2xl": "1rem" },
    },
  },
  plugins: [],
};
