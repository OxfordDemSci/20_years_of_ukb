# Flexible IPC Dendrogram (3 or 4 levels)
# Use this code to replace the existing IPC dendrogram plot in patents_analysis.ipynb
# This version automatically adapts to showing 3 or 4 levels based on available data

import matplotlib.pyplot as plt
import pandas as pd

# Load IPC hierarchy
df_ipc = pd.read_csv('data/patent/ipc_hierarchy.csv')

# Build hierarchy data from topics_list_raw with all levels
ipc_rows = []
for idx, row in df_patent.iterrows():
    topics_raw = row.get('topics_list_raw', [])
    if not isinstance(topics_raw, list):
        continue
    
    for ipc_code in topics_raw:
        if not ipc_code:
            continue
        
        # Initialize all levels
        section_code = None
        section_title = None
        class_code = None
        class_title = None
        subclass_code = None
        subclass_title = None
        maingroup_code = None
        maingroup_title =None
        
        # Try to match exact code first
        matches = df_ipc[df_ipc['symbol'] == ipc_code]
        
        if not matches.empty:
            # Found exact match - get its level and traverse hierarchy
            match = matches.iloc[0]
            path_parts = match['path'].split('/')
            
            # Extract codes from path
            if len(path_parts) > 0:
                section_code = path_parts[0]
                sec_match = df_ipc[(df_ipc['symbol'] == section_code) & (df_ipc['kind'] == 's')]
                section_title = sec_match['title'].iloc[0] if not sec_match.empty else f"Section {section_code}"
            
            if len(path_parts) > 1:
                class_code = path_parts[1]
                cls_match = df_ipc[(df_ipc['symbol'] == class_code) & (df_ipc['kind'] == 'c')]
                class_title = cls_match['title'].iloc[0] if not cls_match.empty else f"Class {class_code}"
            
            if len(path_parts) > 2:
                subclass_code = path_parts[2]
                sub_match = df_ipc[(df_ipc['symbol'] == subclass_code) & (df_ipc['kind'] == 'u')]
                subclass_title = sub_match['title'].iloc[0] if not sub_match.empty else f"Subclass {subclass_code}"
            
            if len(path_parts) > 3:
                maingroup_code = path_parts[3]
                mg_match = df_ipc[(df_ipc['symbol'] == maingroup_code) & (df_ipc['kind'] == 'm')]
                maingroup_title = mg_match['title'].iloc[0] if not mg_match.empty else match['title']
        else:
            # Try to parse structure from code itself
            if len(ipc_code) >= 1:
                section_code = ipc_code[0]
                sec_match = df_ipc[(df_ipc['symbol'] == section_code) & (df_ipc['kind'] == 's')]
                section_title = sec_match['title'].iloc[0] if not sec_match.empty else f"Section {section_code}"
            
            if len(ipc_code) >= 3:
                class_code = ipc_code[:3]
                cls_match = df_ipc[(df_ipc['symbol'] == class_code) & (df_ipc['kind'] == 'c')]
                class_title = cls_match['title'].iloc[0] if not cls_match.empty else f"Class {class_code}"
            
            if len(ipc_code) >= 4:
                subclass_code = ipc_code[:4]
                sub_match = df_ipc[(df_ipc['symbol'] == subclass_code) & (df_ipc['kind'] == 'u')]
                subclass_title = sub_match['title'].iloc[0] if not sub_match.empty else f"Subclass {subclass_code}"
            
            # Main group codes are typically 14 characters (e.g., A01B0001000000)
            if len(ipc_code) >= 10:
                maingroup_code = ipc_code
                mg_match = df_ipc[(df_ipc['symbol'] == maingroup_code) & (df_ipc['kind'] == 'm')]
                maingroup_title = mg_match['title'].iloc[0] if not mg_match.empty else f"Group {ipc_code[-4:]}"
        
        # Only add if we have at least section and one lower level
        if section_code and (class_code or subclass_code or maingroup_code):
            ipc_rows.append({
                'section_code': section_code,
                'section_title': section_title,
                'class_code': class_code,
                'class_title': class_title,
               'subclass_code': subclass_code,
                'subclass_title': subclass_title,
                'maingroup_code': maingroup_code,
                'maingroup_title': maingroup_title
            })

hier_df_ipc = pd.DataFrame(ipc_rows)

if not hier_df_ipc.empty and hier_df_ipc['section_code'].notna().any():
    # Determine which levels to use
    hier_df_4level = hier_df_ipc[
        hier_df_ipc['section_code'].notna() & 
        hier_df_ipc['class_code'].notna() & 
        hier_df_ipc['subclass_code'].notna() & 
        hier_df_ipc['maingroup_code'].notna()
    ].copy()
    
    # Use 4 levels if we have sufficient data, otherwise 3 levels
    if len(hier_df_4level) >= 100:
        hier_df_ipc_complete = hier_df_4level
        use_4_levels = True
        print(f"✓ Using 4-level hierarchy: {len(hier_df_ipc_complete):,} records with Section→Class→Subclass→MainGroup")
    else:
        hier_df_ipc_complete = hier_df_ipc[
            hier_df_ipc['section_code'].notna() & 
            hier_df_ipc['class_code'].notna() & 
            hier_df_ipc['subclass_code'].notna()
        ].copy()
        use_4_levels = False
        print(f"✓ Using 3-level hierarchy: {len(hier_df_ipc_complete):,} records with Section→Class→Subclass")
        print(f"  (Only {len(hier_df_4level)} records had MainGroup data - threshold is 100)")
    
    if not hier_df_ipc_complete.empty:
        # Aggregate counts based on available levels
        if use_4_levels:
            group_cols = ["section_code", "section_title", "class_code", "class_title",
                         "subclass_code", "subclass_title", "maingroup_code", "maingroup_title"]
        else:
            group_cols = ["section_code", "section_title", "class_code", "class_title",
                         "subclass_code", "subclass_title"]
        
        counts = hier_df_ipc_complete.groupby(group_cols).size().reset_index(name="count")
        
        # Filter to top Sections
        TOP_SECTIONS = 5 if use_4_levels else 6
        section_totals = counts.groupby(["section_code", "section_title"])["count"].sum().reset_index()
        section_totals = section_totals.sort_values("count", ascending=False).head(TOP_SECTIONS)
        
        counts = counts.merge(section_totals[["section_code"]], on="section_code", how="inner")
        section_order = section_totals["section_code"].tolist()
        
        # Create figure
        fig_width = 20 if use_4_levels else 16
        fig, ax = plt.subplots(figsize=(fig_width, 20))
        
        # Color scheme
        colors_palette = plt.cm.Set3(range(len(section_order)))
        section_colors = dict(zip(section_order, colors_palette))
        
        # Position parameters
        y_spacing = 2.0
        x_section = 0
        x_class = 4
        x_subclass = 8
        x_maingroup = 12 if use_4_levels else None
        y_current = 0
        
        # Draw each Section and its hierarchy
        for sec_idx, sec_code in enumerate(section_order):
            sec_data = counts[counts["section_code"] == sec_code]
            sec_title = sec_data["section_title"].iloc[0]
            sec_total = sec_data["count"].sum()
            
            # Get top classes for this section
            MAX_CLASSES = 5 if use_4_levels else 6
            class_groups = (
                sec_data.groupby(["class_code", "class_title"])["count"]
                .sum()
                .reset_index()
                .sort_values("count", ascending=False)
                .head(MAX_CLASSES)
            )
            
            class_y_positions = {}
            class_y_start = y_current
            class_spacing = 0.0
            
            # Process each class
            for cls_idx, (_, class_row) in enumerate(class_groups.iterrows()):
                class_code = class_row["class_code"]
                class_title = class_row["class_title"]
                class_total = class_row["count"]
                
                cls_data = sec_data[sec_data["class_code"] == class_code]
                
                # Get top subclasses for this class
                MAX_SUBCLASSES = 4 if use_4_levels else 6
                subclass_groups = (
                    cls_data.groupby(["subclass_code", "subclass_title"])["count"]
                    .sum()
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(MAX_SUBCLASSES)
                )
                
                subclass_y_positions = {}
                subclass_y_start = class_y_start + class_spacing
                
                # Process each subclass
                for sub_idx, (_, subclass_row) in enumerate(subclass_groups.iterrows()):
                    subclass_code = subclass_row["subclass_code"]
                    subclass_title = subclass_row["subclass_title"]
                    subclass_total = subclass_row["count"]
                    
                    if use_4_levels:
                        # Draw with main groups
                        sub_data = cls_data[cls_data["subclass_code"] == subclass_code]
                        
                        MAX_MAINGROUPS = 3
                        maingroup_groups = (
                            sub_data.groupby(["maingroup_code", "maingroup_title"])["count"]
                            .sum()
                            .reset_index()
                            .sort_values("count", ascending=False)
                            .head(MAX_MAINGROUPS)
                        )
                        
                        n_maingroups = len(maingroup_groups)
                        maingroup_spacing = 0.5
                        
                        subclass_y = subclass_y_start + (n_maingroups - 1) * maingroup_spacing / 2
                        subclass_y_positions[subclass_code] = subclass_y
                        
                        # Draw main groups
                        for mg_idx, (_, maingroup_row) in enumerate(maingroup_groups.iterrows()):
                            maingroup_code = maingroup_row["maingroup_code"]
                            maingroup_title = maingroup_row["maingroup_title"]
                            maingroup_count = maingroup_row["count"]
                            
                            maingroup_y = subclass_y_start + mg_idx * maingroup_spacing
                            
                            # Draw connection from subclass to main group
                            ax.plot(
                                [x_subclass, x_maingroup], [subclass_y, maingroup_y],
                                color=section_colors[sec_code], linewidth=1, alpha=0.4, zorder=1
                            )
                            
                            # Draw main group node
                            mg_size = min(400, 100 + maingroup_count * 2)
                            ax.scatter(
                                x_maingroup, maingroup_y, s=mg_size,
                                color=section_colors[sec_code], edgecolor="gray", 
                                linewidth=0.5, zorder=2, alpha=0.4
                            )
                            
                            # Label main group
                            mg_text = maingroup_title[:30] + "..." if len(maingroup_title) > 30 else maingroup_title
                            ax.text(
                                x_maingroup + 0.15, maingroup_y,
                                f"{mg_text} ({maingroup_count})",
                                ha="left", va="center", fontsize=6
                            )
                        
                        subclass_y_start += n_maingroups * maingroup_spacing + 0.3
                    else:
                        # 3-level mode: just position subclass directly
                        subclass_spacing = 0.6
                        subclass_y = subclass_y_start
                        subclass_y_positions[subclass_code] = subclass_y
                        subclass_y_start += subclass_spacing
                    
                    # Draw subclass node
                    sub_size = min(800, 200 + subclass_total * 2)
                    ax.scatter(
                        x_subclass, subclass_y, s=sub_size,
                        color=section_colors[sec_code], edgecolor="gray",
                        linewidth=1, zorder=2, alpha=0.6
                    )
                    
                    # Label subclass
                    sub_text = subclass_title[:25] + "..." if len(subclass_title) > 25 else subclass_title
                    ax.text(
                        x_subclass - 0.15, subclass_y,
                        f"{subclass_code}\n({subclass_total})",
                        ha="right", va="center", fontsize=7
                    )
                
                # Position class at midpoint of its subclasses
                class_y = class_y_start + class_spacing + (subclass_y_start - class_y_start - class_spacing) / 2
                class_y_positions[class_code] = class_y
                
                # Draw connections from class to subclasses
                for subclass_code, subclass_y in subclass_y_positions.items():
                    ax.plot(
                        [x_class, x_subclass], [class_y, subclass_y],
                        color=section_colors[sec_code], linewidth=1.2, alpha=0.5, zorder=1
                    )
                
                # Draw class node
                cls_size = min(1200, 300 + class_total * 2)
                ax.scatter(
                    x_class, class_y, s=cls_size,
                    color=section_colors[sec_code], edgecolor="gray",
                    linewidth=1.5, zorder=2, alpha=0.7
                )
                
                # Label class
                cls_text = class_title[:20] + "..." if len(class_title) > 20 else class_title
                ax.text(
                    x_class - 0.15, class_y,
                    f"{class_code}\n({class_total})",
                    ha="right", va="center", fontsize=8, fontweight="bold"
                )
                
                class_spacing = subclass_y_start - class_y_start
                class_y_start = subclass_y_start + 0.5
            
            # Position section at midpoint of its classes
            sec_y = y_current + (class_y_start - y_current - 0.5) / 2
            
            # Draw connections from section to classes
            for class_code, class_y in class_y_positions.items():
                ax.plot(
                    [x_section, x_class], [sec_y, class_y],
                    color=section_colors[sec_code], linewidth=1.5, alpha=0.6, zorder=1
                )
            
            # Draw section node
            sec_size = min(2000, 500 + sec_total * 2)
            ax.scatter(
                x_section, sec_y, s=sec_size,
                color=section_colors[sec_code], edgecolor="black",
                linewidth=2, zorder=3, alpha=0.8
            )
            
            # Label section
            sec_text = sec_title[:20] + "..." if len(sec_title) > 20 else sec_title
            ax.text(
                x_section - 0.2, sec_y,
                f"{sec_code}\n{sec_text}\n({sec_total})",
                ha="right", va="center", fontsize=10, fontweight="bold"
            )
            
            y_current = class_y_start + y_spacing
        
        # Formatting
        xlim_max = 16 if use_4_levels else 12
        ax.set_xlim(-1.5, xlim_max)
        ax.set_ylim(-1, y_current)
        ax.set_aspect("auto")
        ax.axis("off")
        
        # Add level labels
        ax.text(x_section, -0.5, "Section", ha="center", va="top", fontsize=10, style="italic", fontweight="bold")
        ax.text(x_class, -0.5, "Class", ha="center", va="top", fontsize=10, style="italic", fontweight="bold")
        ax.text(x_subclass, -0.5, "Subclass", ha="center", va="top", fontsize=10, style="italic", fontweight="bold")
        if use_4_levels:
            ax.text(x_maingroup, -0.5, "Main Group", ha="center", va="top", fontsize=10, style="italic", fontweight="bold")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No IPC data with sufficient hierarchy levels available.")
else:
    print("No IPC hierarchy data available for dendrogram plot.")
