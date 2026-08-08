---
layout: leetcode-entry
title: "3302. Find the Lexicographically Smallest Valid Sequence"
permalink: "/leetcode/problem/2026-08-08-3302-find-the-lexicographically-smallest-valid-sequence/"
leetcode_ui: true
entry_slug: "2026-08-08-3302-find-the-lexicographically-smallest-valid-sequence"
---

[3302. Find the Lexicographically Smallest Valid Sequence](https://leetcode.com/problems/find-the-lexicographically-smallest-valid-sequence/solutions/8448787/kotlin-rust-by-samoylenkodmitry-jzkq/) medium
[substack](https://dmitriisamoilenko.substack.com/p/08082026-3302-find-the-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FDcmcobdEC4)

https://dmitrysamoylenko.com/leetcode/

![08.08.2026.webp](/assets/leetcode_daily_images/08.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1445

#### Problem TLDR

Match subsequence with at most one difference

#### Intuition

didn't solved
```j
    // is this kmp algo?
    //
    // all except one - matches
    // vbcca  abc
    // *b c           match the suffix + match the prefix
    //  1 20
    //     0
    // idk - lets' look at hints (8 minute) - dp
    // 25 minute, just gave up
```
*  build the bestpoke first seen subsequence match from the tail, remember the positions
* iterate forward, if match take, if not, see if bestpoke suffix is not intersecting our position, so we can just reuse it

#### Approach

* careful with Rust usize underflow usize-1=usize::MAX

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun validSequence(w: String, t: String) = buildList {
        val last = IntArray(t.length){-1}; var j = t.length-1; var skip = true
        for (i in w.length-1 downTo 0) if (j >= 0 && w[i] == t[j]) last[j--] = i
        j = 0
        for (i in w.indices) {
            if (w[i]==t[j] || (skip && (j==t.length-1||i<last[j+1]))) {
                add(i)
                if (w[i]!=t[j]) skip=false
                j++
                if (j == t.length) break
            }
        }
        if (j!=t.length) clear()
    }
```
```rust
    pub fn valid_sequence(w: String, t: String) -> Vec<i32> {
        let (mut r, mut l) = (vec![], vec![0; t.len()]);let mut j=t.len()-1;
        for i in (0..w.len()).rev() {
            if w.as_bytes()[i] == t.as_bytes()[j] { l[j] = i+1; if j<1 {break};j -= 1 } }
        j = 0; let mut skip = true;
        for i in 0..w.len() {
            if w.as_bytes()[i] == t.as_bytes()[j] || skip && (j+1==t.len()||i+1 < l[j+1]) {
                if w.as_bytes()[i] != t.as_bytes()[j] { skip = false}
                r.push(i as i32); j += 1; if j == t.len() { break }
            }
        } if j == t.len() { r } else { vec![] }
    }
```

