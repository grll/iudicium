# SR-101 XML Document Structure

## Overview
The SR-101-03032024-EN.xml file contains the Swiss Federal Constitution in Akoma Ntoso format, a standard for legal document markup.

## File Characteristics
- **File Size**: ~424KB
- **Format**: Single-line XML (not formatted/pretty-printed)
- **Encoding**: UTF-8
- **Standard**: Akoma Ntoso 3.0
- **Document**: Federal Constitution of 18 April 1999 of the Swiss Confederation
- **Status Date**: March 3, 2024
- **Language**: English translation (non-official)

## XML Structure

### Root Namespace
```xml
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0" 
            xmlns:fedlex="http://fedlex.admin.ch/">
```

### Document Hierarchy
```
akomaNtoso
└── act (name="publicLaw")
    ├── meta
    │   ├── identification
    │   │   ├── FRBRWork (document metadata)
    │   │   ├── FRBRExpression (language version)
    │   │   └── FRBRManifestation (format version)
    │   └── references
    │       ├── TLCOrganization
    │       ├── TLCRole
    │       └── TLCReference
    ├── preface
    │   ├── docNumber (101)
    │   └── docTitle
    ├── preamble
    │   └── Constitutional preamble text
    └── body
        └── title (eId="tit_1" through "tit_6")
            └── article (eId="art_1", "art_2", etc.)
                ├── num (article number with formatting)
                ├── heading (article title)
                └── paragraph (eId="art_X/para_Y")
                    ├── num (paragraph number)
                    └── content
                        └── p (actual text)
```

## Key Elements

### Article Structure
- **Element**: `<article>`
- **ID Pattern**: `eId="art_N"` or `eId="art_N_a"` for sub-articles
- **Example**: `<article eId="art_5_a">`

### Paragraph Structure
- **Element**: `<paragraph>`
- **ID Pattern**: `eId="art_N/para_M"`
- **Contains**: `<num>` for numbering and `<content>` with `<p>` for text

### Text Content
- Most text is wrapped in `<p xmlns:mig="urn:com:c-moria:legi4ch:xslt:migration">`
- Inline formatting: `<b>` (bold), `<i>` (italic), `<sup>` (superscript)
- Special notes: `<authorialNote>` for legislative history references

## Element Frequency (Top Elements)
1. `<num>`: 1084 occurrences (numbering elements)
2. `<b>`: 878 occurrences (bold formatting)
3. `<content>`: 661 occurrences (content containers)
4. `<p>`: 645+ occurrences (paragraph text)
5. `<heading>`: 299 occurrences (section/article headings)
6. `<authorialNote>`: 186 occurrences (legislative annotations)

## Metadata Structure

### FRBRWork (Document Identity)
- Document URI: `https://fedlex.data.admin.ch/eli/cc/1999/404/20240303`
- Entry into force: 2000-01-01
- Document date: 1999-04-18
- Applicability date: 2024-03-03
- SR Number: 101
- Multilingual names (de, fr, en, rm, it)

### Language Support
- German (de): "Bundesverfassung" (BV)
- French (fr): "Constitution fédérale" (Cst.)
- English (en): "Federal Constitution" (Cst.)
- Romansh (rm): "Constituziun federala" (Cst.)
- Italian (it): "Costituzione federale" (Cost.)

## Special Features

### Legislative References
- Extensive use of `<ref>` elements linking to:
  - AS (Amtliche Sammlung) - Official Compilation
  - BBl (Bundesblatt) - Federal Gazette
  - FCD (Federal Council Decree)
  - FedD (Federal Decree)

### Tables
- Some articles contain `<table>` elements with `<tr>` and `<td>` tags
- Used for structured data presentation

### Lists
- `<blockList>` elements for enumerated lists
- `<listIntroduction>` for list preambles
- `<item>` for individual list items

## Namespaces Used
- Default: `http://docs.oasis-open.org/legaldocml/ns/akn/3.0`
- `fedlex`: `http://fedlex.admin.ch/`
- `mig`: `urn:com:c-moria:legi4ch:xslt:migration`
- `data`: `urn:com:c-moria:legi4ch:xslt:data`
- `tmp`: `urn:com:c-moria:legi4ch:xslt:temp`

## Processing Considerations
1. File is a single line - requires XML parser or formatter for readability
2. Large file size (424KB) may require streaming or chunked processing
3. Multiple namespace declarations need proper handling
4. Extensive cross-references require link resolution logic
5. Multilingual content requires language-aware processing