import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import Image
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.lib.styles import ParagraphStyle

# Set custom favicon and page title
st.set_page_config(page_title="Merge table Web App", page_icon="path/logo.jpg")

# Custom CSS to hide Streamlit components
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Initial page
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None  # To store filtered DataFrame
if 'column_headers' not in st.session_state:
    st.session_state.column_headers = None  # To store column headers
if 'column_order' not in st.session_state:
    st.session_state.column_order = None  # To store column order

counter = 1

def insert_column(merged_df, position):
    global counter
    new_column_name = f"id{counter}_0"
    new_column_name2 = f"column_{counter}_1"
    new_column_name3 = f"column_{counter}_2"
    merged_df.insert(position, new_column_name3, merged_df.iloc[:, 2])
    merged_df.insert(position, new_column_name2, merged_df.iloc[:, 1])
    merged_df.insert(position, new_column_name, merged_df.iloc[:, 0])
    counter += 1

def add_page_number(canvas,_,team):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont("Helvetica", 9)
    width, _ = letter

    max_line_width = 100
    x, y = width, 50


    lines = simpleSplit(team, "Helvetica", 9, max_line_width)

    for line in lines:
        canvas.drawString(x, y, line)
        y -= 12
    canvas.drawCentredString((width / 2) + 100, 20, text)

def home_page():
    st.title("Multiple File Upload and Merge by First Column")

    uploaded_dfs = []

    # Upload files
    uploaded_files = st.file_uploader("Upload your CSV or Excel files", type=['csv', 'xlsm', 'xlsx'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read data based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            uploaded_dfs.append(df)
        # Merge files if more than one
        if len(uploaded_dfs) > 1:
            # Start with the first DataFrame
            merged_df = uploaded_dfs[0]
            key_column = merged_df.columns[0]

            # Ensure the key column is string in the first DataFrame
            merged_df[key_column] = merged_df[key_column].astype(str)

            # Iteratively merge the DataFrames
            for df in uploaded_dfs[1:]:
                # Ensure the key column exists and is of string type in the current DataFrame
                if key_column not in df.columns:
                    st.error(f"Key column '{key_column}' not found in one of the uploaded files.")
                    continue

                df[key_column] = df[key_column].astype(str)

                # Merge the DataFrames
                merged_df = pd.merge(merged_df, df, on=key_column, how="inner", validate="1:1")

            # Replace 'nan' with empty strings and clean numeric strings
            merged_df = merged_df.replace('nan', '')
            merged_df.iloc[:, 0] = merged_df.iloc[:, 0].astype(str).str.replace(r'\.0$', '', regex=True)

            # Display the merged DataFrame
            st.write("### Merged Data")
            st.write(merged_df)
        else:
            # Single DataFrame case
            merged_df = uploaded_dfs[0].astype(str)
            merged_df = merged_df.replace('nan', '')
            merged_df.iloc[:, 0] = merged_df.iloc[:, 0].astype(str).str.replace(r'\.0$', '', regex=True)

            # Display the uploaded DataFrame
            st.write("### Uploaded Data")
            st.write(merged_df)

        st.sidebar.image("path/logo.jpg", use_column_width=True)  # For sidebar

        # Filter options
        st.sidebar.header("Filter Options")

        # Dynamic filters
        # Replace 'desired_column' with the actual column you want to filter
        column = 'filter'

        # Filtering logic for a specific column
        if merged_df[column].dtype == 'object':
            filter_value = st.sidebar.selectbox(f"Filter by {column}",
                                                options=['All'] + merged_df[column].unique().tolist())
            filtered_df = merged_df[merged_df[column] == filter_value] if filter_value != 'All' else merged_df
        else:
            min_value, max_value = float(merged_df[column].min()), float(merged_df[column].max())
            filter_range = st.sidebar.slider(f"Filter {column} range:", min_value, max_value, (min_value, max_value))
            filtered_df = merged_df[(merged_df[column] >= filter_range[0]) & (merged_df[column] <= filter_range[1])]

        # Make sure the first row is always included
        filtered_df = pd.concat([merged_df.iloc[[0]], filtered_df]).drop_duplicates().reset_index(drop=True)

        # Save the filtered dataframe to session state
        st.session_state.filtered_df = filtered_df

        # Column selection
        st.sidebar.header("Select Columns to Display")
        selected_columns = st.sidebar.multiselect("Choose columns to display", st.session_state.filtered_df.columns.tolist(), default=st.session_state.filtered_df.columns.tolist())
        st.session_state.column_order = selected_columns

        # Display filtered data
        st.write("### Filtered Data")
        if selected_columns:
            filtered_df_to_display = st.session_state.filtered_df[selected_columns]

            # Display using st_aggrid
            gb = GridOptionsBuilder.from_dataframe(filtered_df_to_display)
            gb.configure_default_column(editable=True, sortable=True, resizable=True)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
            gb.configure_side_bar()
            grid_options = gb.build()

            response = AgGrid(
                filtered_df_to_display,
                gridOptions=grid_options,
                enable_enterprise_modules=False,
                height=500,
                width='100%',
                theme="alpine",
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True
            )

            # Update session_state DataFrame and column order
            if response['data'] is not None:
                st.session_state.filtered_df = pd.DataFrame(response['data'])
            if 'columnState' in response and response['columnState']:
                reordered_columns = [col['colId'] for col in response['columnState'] if 'colId' in col]
                st.session_state.filtered_df = st.session_state.filtered_df[reordered_columns]
                st.session_state.column_order = reordered_columns

            # Download CSV
            # DataFrame
            df = st.session_state.filtered_df

            df = df.fillna(" ")

            df.columns = [col if "Unnamed" not in col else " " for col in df.columns]

            csv_data = df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name='filtered_data.csv',
                mime='text/csv'
            )

            # Export to PDF
            title = st.text_input("Enter a title for the PDF:")
            confidential = st.text_input("Enter a confidential note for the PDF:", value="Confidential")
            team = st.text_input("Enter Create date:")
            confirm = st.text_input("Confirmed by: ")
            Signature_data = st.text_input("Enter Signature data for the PDF:")
            row_page = st.number_input("Enter a number of rows per page:", min_value=1, value=9)
            fontsize = st.number_input("Please enter a font size: ", value=6)

            note1 = st.text_area("Please enter the content (use '-' to separate paragraphs):", height=200)

            if st.button("Generate PDF"):
                # output pdf
                pdf_filename = "output.pdf"

                # Create a BytesIO buffer to save the PDF to memory instead of disk
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=0 * inch,
                                        rightMargin=0 * inch, topMargin=0 * inch, bottomMargin=0 * inch)

                # create story
                story = []
                styles = getSampleStyleSheet()

                # Logo
                logo_path = 'path/logo.jpg'  # logo location
                logo = Image(logo_path, width=2 * inch, height=1 * inch)


                text_line1 = ""
                text_paragraph1 = Paragraph(text_line1, styles['Normal'])
                text_paragraph2 = Paragraph(confidential, styles['Normal'])
                spacer = Spacer(0, 0.35 * inch)

                # Combine text
                text_block = [text_paragraph1, spacer, text_paragraph2]
                table_data = [[logo,"", text_block]]

                # Create table layout
                table1 = Table(table_data, colWidths=[8 * inch, None])
                table1.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 1 * mm),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0 * mm),
                ]))

                new_columns = []
                modified_columns = []



                for i, col in enumerate(st.session_state.filtered_df.columns):
                    if 'Unnamed' in col and i > 0:
                        new_columns.append(new_columns[i - 1])
                        modified_columns.append(i + 1)
                    else:
                        new_columns.append(col)


                st.session_state.filtered_df.columns = new_columns


                new_header = st.session_state.filtered_df.columns.tolist()
                st.session_state.filtered_df.loc[-1] = new_header
                st.session_state.filtered_df.index = st.session_state.filtered_df.index.astype(int) + 1

                st.session_state.filtered_df = st.session_state.filtered_df.sort_index()  # sorts the index



                # Define margins and usable width
                left_margin = 0.1 * inch
                right_margin = 0.1 * inch
                usable_width = landscape(letter)[0] - left_margin - right_margin  # Total width minus margins

                col_widths = []
                for col in st.session_state.filtered_df.columns:
                    len_num = len(col) * 0.08 * inch
                    len_num = max(50, min(120, len_num))
                    col_widths.append(len_num)
                # Check if col_widths is empty
                if not col_widths:
                    print("Column widths are empty.")
                    return None  # Early return if column widths are not determined

                # Initialize variables
                total_col_width = 0  # To track the total width of columns on the current page
                max_cols_per_page = 0  # To count the maximum columns per page
                p = 0  # Counter for pages
                max1 = []  # List to store the maximum number of columns per page


                # Iterate through column widths
                for width in col_widths:
                    total_col_width += width  # Add the current column width to the total

                    # Check if the total width is within the usable width
                    if total_col_width <= usable_width:
                        max_cols_per_page += 1  # Increment the count of columns on the current page
                        p += 1  # Increment the page counter
                    else:
                        # If the width exceeds usable space, finalize the current page
                        max1.append(max_cols_per_page)  # Store the count of columns for the current page

                            # Insert a column for the current page in the DataFrame
                        insert_column(st.session_state.filtered_df, p)

                            # Prepare for the next page
                        p += 4  # Increment the page counter by 4
                        total_col_width = width + sum(col_widths[:3])  # Reset the total column width
                        max_cols_per_page = 4  # Reset the maximum columns per page

                # Append the last page's column count
                max1.append(max_cols_per_page)

                data = [[str(item) for item in row] for row in ([
                                                                    st.session_state.filtered_df.columns.tolist()] + st.session_state.filtered_df.values.tolist())]

                data = data[1:]

                col_widths = []
                for col in st.session_state.filtered_df.columns:

                    len_num = len(col) * 0.08 * inch
                    len_num = max(50, min(120, len_num))

                    col_widths.append(len_num)

                if not col_widths:
                    print("Column widths are empty.")
                    return None  # Early return if column widths are not determined

                # Ensure we have columns to display
                if max_cols_per_page == 0:
                    print("No columns fit on the page.")
                    return None  # Early return if no columns can be displayed

                # Define row handling logic as before
                # Constants and initial setup
                max_rows_per_page = row_page + 2
                rows = len(data)
                num_row_pages = (rows // max_rows_per_page) + 1
                num_col_pages = len(max1)

                start_row = 0

                # Iterate over row pages
                for row_page in range(num_row_pages):
                    # Determine row range for the current page
                    start_row = 1 if start_row == 0 else end_row - 1
                    end_row = start_row + max_rows_per_page - 1

                    end_col = 0

                    # Iterate over column pages
                    for col_page in range(num_col_pages):
                        # Determine column range for the current page
                        start_col = end_col
                        end_col = end_col + max1[col_page]

                        # Prepare page data and create the table
                        page_data = [row[start_col:end_col] for row in data[start_row + 1:end_row]]

                        # Extract two header rows from data and insert them into page_data
                        page_data.insert(0, data[0][start_col:end_col])  # Insert the first header row
                        page_data.insert(1, data[1][start_col:end_col])  # Insert the second header row

                        # Skip if page_data is empty
                        if not page_data or not page_data[0]:
                            print("Page data is empty.")
                            continue

                        # Add the title to the story
                        story.append(table1)
                        styles['Title'].fontSize = 12
                        titles = Paragraph(title, styles['Title'])
                        story.append(titles)

                        style_header = ParagraphStyle(name="HeaderStyle", fontSize=fontsize + 2, leading=fontsize + 5,
                                                      textColor="white", fontName="Helvetica-Bold")
                        style_body = ParagraphStyle(name="BodyStyle", fontSize=fontsize, leading=fontsize + 3,
                                                    textColor="black")

                        # Wrap table data into Paragraphs with conditional styles
                        wrapped_data = [
                            [
                                Paragraph(str(item), style_header if row_idx < 2 else style_body)
                                for item in row
                            ]
                            for row_idx, row in enumerate(page_data)
                        ]


                        # Create table and apply styles
                        table = Table(wrapped_data, colWidths=col_widths[start_col:end_col])
                        table_style = TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 1), colors.red),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('TOPPADDING', (0, 0), (-1, -1), 1 * mm),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 0 * mm),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ])

                        col_idx = 0
                        while col_idx < len(page_data[0]):
                            start_idx = col_idx
                            # Find the continuous range of identical column headers in row 0.
                            while (col_idx + 1 < len(page_data[0]) and page_data[0][col_idx] == page_data[0][col_idx + 1]):
                                col_idx += 1

                            # If there are identical consecutive column headers, apply merging.
                            if start_idx != col_idx:
                                # use SPAN
                                table_style.add('SPAN', (start_idx, 0), (col_idx, 0))
                                # Clear the duplicate column headers, keeping only the first one.
                                for i in range(start_idx + 1, col_idx + 1):
                                    wrapped_data[0][i] = Paragraph("", style_header)

                            col_idx += 1

                        for col_idx in range(len(page_data[1])):
                            if page_data[1][col_idx] == "":
                                # If the value in row 1 is empty, merge row 0 and row 1.
                                table_style.add('SPAN', (col_idx, 0), (col_idx, 1))

                        # use style
                        table.setStyle(table_style)

                        story.append(table)

                        # Add page break if not the last row or column page
                        if row_page < num_row_pages - 1 or col_page < num_col_pages - 1:
                            story.append(PageBreak())

                # Left and right content

                left_content = [
                    Paragraph(f"Confirmed by ({confirm}):", styles['Normal']),
                    Spacer(0, 14),
                    Paragraph("Signature: ________________", styles['Normal']),
                    Paragraph(f"Date: {Signature_data}"),
                ]

                # Split user input note1 by '-' symbol and remove extra spaces
                note_lines = [line.strip() for line in note1.split('-') if line.strip()]

                styles['Normal'].fontSize = 10
                styles['Normal'].leading = 12

                # Convert each line into Paragraph objects
                note_paragraphs = [Paragraph(f"- {line}" if i > 0 else line, styles['Normal']) for i, line in
                                   enumerate(note_lines)]

                right_content = [
                    *note_paragraphs,
                    Paragraph("", styles['Normal']),
                    Paragraph("", styles['Normal']),

                ]

                # Combine left and right content into a table
                combined_data = [[
                    Table([[line] for line in left_content], colWidths=[5 * inch]),"",
                    Table([[line] for line in right_content], colWidths=[4.5 * inch]),
                ]]

                combined_table = Table(combined_data, colWidths=[1 * inch, 4.5 * inch])
                combined_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))

                story.append(combined_table)

                # Build the PDF and save to buffer
                doc.build(story, onFirstPage=lambda canvas, _: add_page_number(canvas, _, team),
                          onLaterPages=lambda canvas, _: add_page_number(canvas, _, team))

                now = datetime.now()
                today_date = now.strftime('%Y-%m-%d_%H-%M-%S')
                milliseconds = now.strftime('%f')[:3]
                pdf_filename1 = f"{pdf_filename}_{today_date}_{milliseconds}.pdf"


                system_pdf_path = os.path.join("E:", "Merge",
                                               pdf_filename1)
                os.makedirs(os.path.dirname(system_pdf_path), exist_ok=True)

                buffer.seek(0)
                with open(system_pdf_path, 'wb') as f:
                    f.write(buffer.read())

                # Allow user to download the PDF
                buffer.seek(0)
                st.download_button(
                    label="Download the generated PDF",
                    data=buffer,
                    file_name=pdf_filename,
                    mime='application/pdf'
                )

        else:
            st.write("Please select at least one column to display.")
    else:
        st.write("Please upload files to start merging and filtering data.")


if st.session_state.page == 'home':
    home_page()
