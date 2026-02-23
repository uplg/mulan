import {
  Plus,
  Home,
  FolderOpen,
  Music,
  Search,
  Settings,
} from "lucide-react";
import { useApp } from "@/lib/store";
import type { AppState } from "@/lib/store";
import { cn } from "@/lib/utils";

const navItems: { icon: React.ElementType; label: string; view: AppState["view"] }[] = [
  { icon: Plus, label: "Create", view: "create" },
  { icon: Home, label: "Home", view: "home" },
  { icon: FolderOpen, label: "Studio", view: "studio" },
  { icon: Music, label: "Library", view: "library" },
  { icon: Search, label: "Search", view: "search" },
  { icon: Settings, label: "Settings", view: "settings" },
];

export function Sidebar() {
  const { state, dispatch } = useApp();

  return (
    <aside className="flex w-45 min-w-45 flex-col border-r border-border bg-sidebar p-4">
      <div className="px-3 pb-6 text-[26px] font-extrabold tracking-tight text-foreground">
        MuLa(n)
      </div>

      <nav className="flex flex-col gap-0.5">
        {navItems.map(({ icon: Icon, label, view }) => (
          <button
            key={view}
            onClick={() => dispatch({ type: "SET_VIEW", view })}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
              state.view === view
                ? "bg-sidebar-accent text-foreground"
                : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
            )}
          >
            <Icon className="h-5 w-5" />
            {label}
          </button>
        ))}
      </nav>
    </aside>
  );
}
