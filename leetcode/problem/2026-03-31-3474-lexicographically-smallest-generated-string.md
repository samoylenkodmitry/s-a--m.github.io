---
layout: leetcode-entry
title: "3474. Lexicographically Smallest Generated String"
permalink: "/leetcode/problem/2026-03-31-3474-lexicographically-smallest-generated-string/"
leetcode_ui: true
entry_slug: "2026-03-31-3474-lexicographically-smallest-generated-string"
---

[3474. Lexicographically Smallest Generated String](https://open.substack.com/pub/dmitriisamoilenko/p/31032026-3474-lexicographically-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard

[youtube](https://youtu.be/uEPNgsYGgJc)

![31.03.2026.webp](/assets/leetcode_daily_images/31.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1314

#### Problem TLDR

String from pattern matches #hard #greedy #kmp

#### Intuition

Didn't solve.
```j
    // b
    // FFFF - aaaa
    // TFFF   baaa    F - always a
    //                T - always str2
    // 10^4 * 500 = 10^6 can be accepted
    // F - is not always 'a', if str2 can match

    // TFFF   aa
    // aa
    //  ab
    //    ab       the matching part is simple
    //             the non-matching: maybe increment it? aa-ab-ac-..-az-ba
    // maybe FFF..s create a pattern?
    //       aba
    // TFT
    // ababa
```
Greedy n*m solution:
1. fill by T
2. validate by T
3. validate by F
4. for matching F's: increment rightmost letter

#### Approach

* we can initialize with 'a'
* we can compare with 'a' to skip hold positions

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(n+m)$$

#### Code

```kotlin
// 80ms
    fun generateString(a: String, b: String): String {
        val s = CharArray(a.length+b.length-1) { 'a' }
        for (i in a.indices) if (a[i] == 'T') for (j in b.indices) s[i+j] = b[j]
        for (i in a.indices) if (a[i] == 'F' && b.indices.all { s[i+it] == b[it]})
            ++s[i+(b.indices.findLast { s[i+it] == 'a' } ?: return "")]
            else if (a[i] == 'T' && b.indices.any { s[i+it] != b[it]}) return ""
        return String(s)
    }
```
```rust
// 3ms
    pub fn generate_string(a: String, b: String) -> String {
        let (a, b, mut s) = (a.as_bytes(), b.as_bytes(), vec![b'a'; a.len() + b.len() - 1]);
        for i in 0..a.len() { if a[i] == b'T' { s[i..i+b.len()].copy_from_slice(b) } }
        for i in 0..a.len() { if a[i] == b'F' && &s[i..i+b.len()] == b {
                let Some(j) = s[i..i+b.len()].iter().rposition(|&c| c == b'a') else { return "".into() };
                s[i+j] += 1
            } else if a[i] == b'T' && &s[i..i+b.len()] != b { return "".into() }}
        String::from_utf8(s).unwrap()
    }
```

