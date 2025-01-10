import streamlit as st
import fitz
from PIL import Image
import base64
import pandas as pd
import pdfplumber
import json
import re
from io import StringIO
from openai import OpenAI
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(
    page_title="PDF Table Extractor",
    # page_icon="📄",
    layout="wide"  # Use the wide layout
)

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

FORMAT1 = {
    "type": "json_schema",
    "json_schema": {
        "name": "list_of_dicts_response",
        "schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True
                    }
                }
            },
            "required": ["data"],
            "additionalProperties": False
        },
        "strict": False
    }
}

def pdf_viewer(input, width=700, height=800):
    # Convert binary data to base64 for embedding
    base64_pdf = base64.b64encode(input).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" style="border:none;"></iframe>'
    st.components.v1.html(pdf_display, height=height + 50)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_section_as_image(pdf_path, page_num, output_image_path):
    rect = (0, 0, 792, 612)
    pdf = fitz.open(pdf_path)
    page = pdf[page_num]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cropped_img = img.crop(rect)
    cropped_img.save(output_image_path)
    pdf.close()
    return output_image_path

def clean_data(input_string):
    cleaned_string = re.sub(r"[`~]", "", input_string)
    return cleaned_string

def give_column_names_rows(base64_img, columns):
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"Return JSON document with data (column names and 5 rows of the table). Only return JSON not other text. The dataset has {columns} columns"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{base64_img}"}
                    },
                ]
            },
        ],
        response_format=FORMAT1,
        max_tokens=500,
    )

    output = response1.choices[0].message.content.strip()
    output_dict = json.loads(output)
    data = output_dict["data"]
    column_names = list(data[0].keys())
    rows = [list(entry.values()) for entry in data]
    print(column_names)
    print(rows)
    # column_names = ['Resident Name', 'Order Description', 'Order Directions', 'Order Category', 'Start Date', 'Indications for Use']
    # rows = [['BURROLA, ADIS (MH100797)', 'Aplisol Solution 5 UNIT/0.1ML (Tuberculin PPD)', 'Inject 0.1 ml intradermally in the evening every 356 day(s) for TB Screening Every evening within annually. Record result within 72 hours. Perform Chest X-Ray if patient refused', 'Pharmacy', '03/02/2024', 'TB Screening'], ['BURROLA, ADIS (MH100797)', 'Atorvastatin Calcium Oral Tablet 20 MG (Atorvastatin Calcium)', 'Give 1 tablet by mouth at bedtime for hyperlipidemia', 'Pharmacy', '07/19/2024', 'hyperlipidemia'], ['BURROLA, ADIS (MH100797)', 'Cholecalciferol Tablet 1000 UNIT', 'Give 2 tablet by mouth one time a day for supplement', 'Pharmacy', '08/31/2024', 'supplement'], ['BURROLA, ADIS (MH100797)', 'hydraLAZINE HCl Oral Tablet 25 MG (Hydralazine HCl)', 'Give 1 tablet by mouth every 6 hours as needed for SBP>170.', 'Pharmacy', '03/01/2024', 'SBP>170.'], ['CAMPELL, ROBERT B (MH100302)', 'AmLODIPine Besylate Tablet 5 MG', 'Give 1 tablet by mouth one time a day for HTN Hold for SBP <100 or HR <60', 'Pharmacy', '10/18/2024', 'HTN']]

    return column_names, rows

def generate_final_csv(pdf_path, starting_page, ending_page, column_names, rows):
    page_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[starting_page - 1:ending_page]:
            page_text += page.extract_text()
            page_text += "\n"

    prompt = f"""
        Please extract the data as csv from the given PDF (where tables span across pages). Ensure that each field is transcribed word-for-word exactly as it appears in the document without adding or modifying any information. Keep the original order of rows and columns and include every row of each page. Please ensure that all values, even blank cells, match the original document exactly. Please do not miss any rows in any page of the document.
        The column names and 5 rows are provided to understand the schema. Separate each cell by pipe (|). Analyze the table schema, including column names and data patterns, and extract all additional rows such that they match the provided structure, ensuring consistency with the column types and formatting.
        Do not include 'Certainly! Below is the extracted data from the PDF'. Answer should be directly readable as csv.
        Text:
        {page_text}
        """
    example_prompt = f"""
        Input:
        {page_text[:1000]}

        Output:
        Column Names: {column_names}
        Row1: {rows[0]}
        Row2: {rows[1]}
        Row3: {rows[2]}
        Row4: {rows[3]}
        Row5: {rows[4]}
        """

    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates csv text from pdf"},
            {"role": "system", "content": example_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    # Step 4: Print response
    table_output = response2.choices[0].message.content
    table_output = clean_data(table_output)
    # table_output = '''
    # Diagnosis|Onset|Resolved|Rank|Classification as of Date
    # MUSCLE WEAKNESS (GENERALIZED) (M62.81)|6/13/24||Diagnosis #5|Medicare Part B
    # NEED FOR ASSISTANCE WITH PERSONAL CARE (Z74.1)|9/19/23|6/13/24|Diagnosis #5|Medicare Part B
    # NEED FOR ASSISTANCE WITH PERSONAL CARE (Z74.1)|6/21/23|6/21/23|Diagnosis-Other|Medicare Part B
    # OTHER ABNORMALITIES OF GAIT AND MOBILITY (R26.89)|6/21/23|9/1/23|Diagnosis-Other|Medicare Part B
    # NEED FOR ASSISTANCE WITH PERSONAL CARE (Z74.1)|3/9/23|3/9/23|Diagnosis-Other|Medicare Part B
    # OTHER ABNORMALITIES OF GAIT AND MOBILITY (R26.89)|3/9/23|3/9/23|Diagnosis-Other|Medicare Part B
    # NEED FOR ASSISTANCE WITH PERSONAL CARE (Z74.1)|12/17/22|1/24/23|Diagnosis #5|During Stay
    # OTHER ABNORMALITIES OF GAIT AND MOBILITY (R26.89)|12/17/22|1/24/23|Diagnosis #6|During Stay
    # COVID-19 (U07.1)|12/16/22|1/31/23|Diagnosis (#67)|(#69)
    # ABNORMAL POSTURE (R29.3)|9/15/22|9/30/22|Diagnosis-Other|During Stay
    # DIFFICULTY IN WALKING, NOT ELSEWHERE CLASSIFIED (R26.2)|9/15/22|9/30/22|Diagnosis-Other|During Stay
    # ABNORMAL POSTURE (R29.3)|3/16/22|5/31/22|Diagnosis #6|During Stay
    # ABNORMAL POSTURE (R29.3)|3/16/22|6/6/22|Diagnosis-Other|During Stay
    # OTHER REDUCED MOBILITY (Z74.09)|3/16/22|5/31/22|Diagnosis #5|During Stay
    # OTHER REDUCED MOBILITY (Z74.09)|3/16/22|6/14/22|Diagnosis-Other|During Stay
    # OTHER ABNORMALITIES OF GAIT AND MOBILITY (R26.89)|1/28/22|1/28/22|Diagnosis #7|During Stay
    # ACUTE KIDNEY FAILURE, UNSPECIFIED (N17.9)|11/9/21||Diagnosis #8|Admission
    # BIPOLAR DISORDER, UNSPECIFIED (F31.9)|11/9/21||Diagnosis #12|Admission
    # CALCULUS OF KIDNEY WITH CALCULUS OF URETER (N20.2)|11/9/21||Diagnosis #9|Admission
    # CHRONIC VENOUS HYPERTENSION (IDIOPATHIC) WITH ULCER OF LEFT LOWER EXTREMITY (I87.312)|11/9/21||Diagnosis #15|Admission
    # DEVELOPMENTAL DISORDER OF SCHOLASTIC SKILLS, UNSPECIFIED (F81.9)|11/9/21||Diagnosis #13|Admission
    # ESSENTIAL (PRIMARY) HYPERTENSION (I10)|11/9/21||Diagnosis #14|Admission
    # LYMPHEDEMA, NOT ELSEWHERE CLASSIFIED (I89.0)|11/9/21||Diagnosis #10|Admission
    # MORBID (SEVERE) OBESITY DUE TO EXCESS CALORIES (E66.01)|11/9/21||Diagnosis #4|Admission
    # MUSCLE WEAKNESS (GENERALIZED) (M62.81)|11/9/21|12/29/21|Diagnosis #5|Admission
    # OTHER REDUCED MOBILITY (Z74.09)|11/9/21|12/7/21|Diagnosis #6|Admission
    # SCHIZOAFFECTIVE DISORDER, BIPOLAR TYPE (F25.0)|11/9/21||Diagnosis #11|Admission
    # TYPE 2 DIABETES MELLITUS WITH DIABETIC NEPHROPATHY (E11.21)|11/9/21||Diagnosis #3|Admission
    # TYPE 2 DIABETES MELLITUS WITH OTHER SPECIFIED COMPLICATION (E11.69)|11/9/21||Admission|Diagnosis (#67)
    # UNSPECIFIED DIASTOLIC (CONGESTIVE) HEART FAILURE (I50.30)|11/9/21||Diagnosis #2|Admission
    # MUSCLE WEAKNESS (GENERALIZED) (M62.81)|9/14/21|11/6/21|Diagnosis #4|During Stay
    # OTHER REDUCED MOBILITY (Z74.09)|9/14/21|11/6/21|Diagnosis #3|During Stay
    # SCHIZOAFFECTIVE DISORDER, BIPOLAR TYPE (F25.0)|7/26/21|11/6/21|Diagnosis #16|During Stay
    # ACUTE RESPIRATORY FAILURE, UNSPECIFIED WHETHER WITH HYPOXIA OR HYPERCAPNIA (J96.00)|7/21/21|11/6/21|Diagnosis #7|Admission
    # BIPOLAR DISORDER, UNSPECIFIED (F31.9)|7/21/21|11/6/21|Diagnosis #14|Admission
    # CHRONIC KIDNEY DISEASE, UNSPECIFIED (N18.9)|7/21/21|11/6/21|Diagnosis #9|Admission
    # DEVELOPMENTAL DISORDER OF SCHOLASTIC SKILLS, UNSPECIFIED (F81.9)|7/21/21|11/6/21|Diagnosis #13|Admission
    # DYSPHAGIA, ORAL PHASE (R13.11)|7/21/21|11/6/21|Diagnosis #5|Admission
    # ESSENTIAL (PRIMARY) HYPERTENSION (I10)|7/21/21|11/6/21|Diagnosis #11|Admission
    # IRON DEFICIENCY ANEMIA, UNSPECIFIED (D50.9)|7/21/21|11/6/21|Diagnosis #11|Admission
    # LYMPHEDEMA, NOT ELSEWHERE CLASSIFIED (I89.0)|7/21/21|11/6/21|Diagnosis #6|Admission
    # '''
    return table_output


if "column_names" not in st.session_state:
    st.session_state.column_names = []
if "rows" not in st.session_state:
    st.session_state.rows = []
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False  # Tracks if table data is generated
if "preview_ready" not in st.session_state:
    st.session_state.preview_ready = False  # Tracks if the preview is ready

# Title
st.title("PDF Table Extractor")
base64_img = None
# File Upload
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
if uploaded_file is not None:
    binary_data = uploaded_file.getvalue()
    st.write("Preview of the uploaded PDF:")
    pdf_viewer(input=binary_data)

    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded successfully!")

    # User Inputs
    start_page = st.number_input("Starting Page", min_value=1, step=1)
    end_page = st.number_input("Ending Page", min_value=start_page, step=1)
    num_columns = st.number_input("Number of Columns", min_value=1, step=1)

    if start_page and end_page and num_columns and st.button("Next"):
        output_image_path = "cropped_section.png"

        extract_section_as_image(pdf_path, start_page - 1, output_image_path)

        base64_img = f"data:image/png;base64,{encode_image(output_image_path)}"

        st.session_state.column_names, st.session_state.rows = give_column_names_rows(
            base64_img, num_columns
        )
        st.session_state.preview_ready = True  # Mark preview as ready

    # Display Columns and Rows if preview is ready
    if st.session_state.preview_ready:
        st.subheader("Preview Table")

        # Create a DataFrame for columns and rows
        row_df = pd.DataFrame(st.session_state.rows, columns=st.session_state.column_names)

        # Calculate height dynamically based on the number of rows
        grid_height = len(row_df) * 35 + 50  # Adjust row height and header height
        if grid_height > 400:
            grid_height = 400  # Set a maximum height limit

        # Use AgGrid for editable cells
        # st.markdown("### Editable Table")
        gb = GridOptionsBuilder.from_dataframe(row_df)
        gb.configure_default_column(editable=True)  # Make all cells editable
        gb.configure_grid_options(domLayout='autoHeight')  # Automatically adjust grid height
        grid_options = gb.build()

        response = AgGrid(
            row_df,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode='VALUE_CHANGED',  # Detect changes in cell values
            fit_columns_on_grid_load=True,  # Disable auto-fit for columns
            height=grid_height,  # Dynamically calculated height
            reload_data=False,
        )

        # Get the updated DataFrame after edits
        updated_data = response['data']
        st.session_state.rows = updated_data.values.tolist()  # Update session state with new rows

        col_empty1, col_empty2, col_buttons = st.columns([7, 1, 2])
        with col_buttons:
            col1, col2 = st.columns(2)
            regenerate_clicked = col1.button("Regenerate")
            next_clicked = col2.button("Generate CSV")

        if regenerate_clicked:
            # Regenerate column names and rows
            st.session_state.column_names, st.session_state.rows = give_column_names_rows(
                base64_img, num_columns
            )
            st.rerun()  # Refresh the app to show regenerated data

        if next_clicked:
            # Process and generate final CSV
            # st.write("Processing data from the PDF...")
            table_output = generate_final_csv(pdf_path, start_page, end_page, st.session_state.column_names, st.session_state.rows)
            data_io = StringIO(table_output)
            df = pd.read_csv(data_io, delimiter="|", on_bad_lines="skip")

            # Display CSV content
            st.subheader("Generated CSV")
            st.dataframe(df, use_container_width=True)

            # # Provide download button
            # st.download_button(
            #     label="Download CSV",
            #     data=df.to_csv(index=False),
            #     file_name="extracted_table.csv",
            #     mime="text/csv",
            # )
