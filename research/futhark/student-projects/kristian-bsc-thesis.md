# Extracting Certified Futhark Code from Coq

## Metadata
- **Authors:** K R I S T I A N K N U D S E N O L E S E N
- **Venue/Year:** Bachelor Project in Computer Science, Department of Computer Science, University of Copenhagen, November 1, 2021

## Summary
This bachelor's project investigates the feasibility of using the MetaCoq and ConCert frameworks to extract certified Futhark code from the Coq proof assistant. The author extends the FuCert framework to implement and extract the maximum segment sum function while proving its functional correctness. The project explores how to model Futhark arrays in Coq, handle the monoid criteria required by Futhark's reduce and scan operations, and ensure that Coq's guarantees carry over to the extracted Futhark code. The work demonstrates that it is viable to generate certified Futhark code using this approach, though several technical challenges must be addressed, particularly around array modeling, equality handling, and extraction limitations.

## Key Contributions
- Extended the FuCert framework with infrastructure for modeling Futhark arrays in Coq
- Implemented the maximum segment sum function with formal verification of associativity and functional correctness
- Developed custom automation and tactics to handle subset types and proof irrelevance
- Demonstrated the viability of extracting certified Futhark code from Coq proofs
- Created infrastructure for specifying and implementing Futhark functions in Coq with guaranteed properties

## Technical Approach
The project uses Coq's subset types to model Futhark arrays with size parameters, implementing induction principles and custom tactics to work around equality complications. The author creates module types and functors to specify Futhark functions in Coq, ensuring that remapped functions share the same properties as their Futhark counterparts. For the maximum segment sum example, the author proves associativity of a custom operator using subset types with decidable equality predicates, and verifies functional correctness using segment definitions. The extraction process uses MetaCoq's erasure procedure and ConCert's type information to generate Futhark code, though limitations exist around polymorphic types and size parameters.

## Results
The project successfully implements and extracts the maximum segment sum function to Futhark, with formal proofs of associativity and functional correctness in Coq. The extracted code is verified to be correct when no overflow occurs, though the bounded precision of Futhark's integers means guarantees only hold under this condition. The author develops comprehensive infrastructure including array models, induction principles, and automation tactics that make the framework practical to use. The work demonstrates that while challenges exist, the approach is viable for generating certified Futhark code.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work as it explores the intersection of formal verification and high-performance parallel programming. The techniques developed for modeling arrays, handling equality, and extracting code could inform the design of other verified compilation pipelines. The work also addresses practical concerns in parallel programming, such as verifying monoid properties for parallel operations and ensuring array bounds safety. The approach of using proof assistants to generate certified high-performance code has implications for building trustworthy systems in domains where correctness is critical.
