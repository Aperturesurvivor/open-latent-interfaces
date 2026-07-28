# Phi 3.5 Mini audited operand reader v1

This release publishes the first audited latent read bridge from Open Latent
Interfaces.

For `microsoft/Phi-3.5-mini-instruct` at revision
`2fe192450127e6a83f7441aef6e3ca586c338b77`, the reader decodes decimal
operand digits from residual hidden-state index 1 using ten frozen full-width
native-state centroids.

- Selection: 180/180 exact operand pairs
- Development: 45/45 exact operand pairs
- Sealed audit: 45/45 exact operand pairs
- Rotated-label selection control: 0/180 exact pairs
- Residual width: 3,072
- Tensor SHA-256:
  `58f84aeda73713e9eb2e8ed0347639fc84f60273ad69557d7718b096cd6ac0c0`

An external semantic locator supplies the two operand token spans and verifies
one token per decimal character. Autonomous operand discovery is not claimed.

The containing read-compute-write audit missed its overall uplift gate because
the answer writer generalized imperfectly; the reader component independently
passed. The release contains only the reader tensors.
