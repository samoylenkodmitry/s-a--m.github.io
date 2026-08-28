---
layout: leetcode-entry
title: "3734. Lexicographically Smallest Palindromic Permutation Greater Than Target"
permalink: "/leetcode/problem/2026-08-28-3734-lexicographically-smallest-palindromic-permutation-greater-than-target/"
leetcode_ui: true
entry_slug: "2026-08-28-3734-lexicographically-smallest-palindromic-permutation-greater-than-target"
---

[3734. Lexicographically Smallest Palindromic Permutation Greater Than Target](https://leetcode.com/problems/lexicographically-smallest-palindromic-permutation-greater-than-target/solutions/8487352/kotlin-rust-by-samoylenkodmitry-5ksc/) hard
[substack](https://dmitriisamoilenko.substack.com/p/28082026-3734-lexicographically-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Tx6xmCM36PI)

https://dmitrysamoylenko.com/leetcode/

![28.08.2026.webp](/assets/leetcode_daily_images/28.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1465

#### Problem TLDR

Smallest permutation palindrome bigger than target

#### Intuition

Try to one-up every position. The rest is a buildup of a palindrome by using the frequencies.

#### Approach

* to match the prefix adjust the frequency; exit at the first mismatch
* half the frequencies

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun lexPalindromicPermutation(s: String, t: String): String {
        val h = IntArray(26); for (c in s) ++h[c-'a']
        if (h.count { it % 2 > 0 } > 1) return ""
        val m = (0..25).find { h[it] % 2 > 0 }?.let { 'a' + it }
        var p = ""; var ans = ""; for (c in 0..25)h[c] /= 2
        fun b(p: String) = (0..25).fold(p) { a, i -> a + "${'a' + i}".repeat(h[i]) }.let { it + (m ?: "") + it.reversed() }
        for (c in t.take(t.length / 2)) {
            (c - 'a' + 1..25).find { h[it] > 0 }?.let { x -> h[x]--; ans = b(p + ('a' + x)); h[x]++ }
            if (h[c - 'a']-- == 0) return ans; p += c
        }
        return b(p).takeIf { it > t } ?: ans
    }
```
```rust
    pub fn lex_palindromic_permutation(s: String, t: String) -> String {
        let mut f = [0usize; 123]; for b in s.bytes() { f[b as usize] += 1 }
        if f.iter().map(|x| x & 1).sum::<usize>() > 1 { return "".into() }
        let m: String = (97u8..123).filter(|&i| f[i as usize] & 1 > 0).map(|i| i as char).collect();
        for x in &mut f { *x /= 2 }; let (mut p, mut ans) = (String::new(), String::new());
        let k = |p: &str, f: &[usize]| {
            let h = format!("{p}{}", (97u8..123).flat_map(|i| vec![i as char; f[i as usize]]).join(""));
            format!("{h}{m}{}", h.chars().rev().join(""))
        };
        for b in t[..t.len() / 2].bytes() {
            (b + 1..123u8).find(|&c| f[c as usize] > 0).map(|c| {
                f[c as usize] -= 1; p.push(c as char); ans = k(&p, &f); p.pop(); f[c as usize] += 1
            });
            if f[b as usize] == 0 { return ans }; f[b as usize] -= 1; p.push(b as char)
        }
        let r = k(&p, &f); if r > t { r } else { ans }
    }
```

