---
layout: leetcode-entry
title: "1081. Smallest Subsequence of Distinct Characters"
permalink: "/leetcode/problem/2026-07-19-1081-smallest-subsequence-of-distinct-characters/"
leetcode_ui: true
entry_slug: "2026-07-19-1081-smallest-subsequence-of-distinct-characters"
---

[1081. Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/solutions/8407086/kotlin-rust-by-samoylenkodmitry-n5q1/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19072026-1081-smallest-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NoN0DU7Wk1U)

https://dmitrysamoylenko.com/leetcode/

![19.07.2026.webp](/assets/leetcode_daily_images/19.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1425

#### Problem TLDR

Smallest uniq-chars subsequence

#### Intuition

Didn't solve. Stack-based algorightm was a little bit unexpected for me.
```j
  // 46 minute TLE, n^3 algo, full search dp
    // 26^26 = 6*10^36
    // basically i give up at 1:28
    // the solution is stack-based
    // remove from stack if we find a better variant and stack peek
    // can be found forward
```
Greedily take every char.
If current is better than the last taken, consider popping the last if there is the same in the suffix.

#### Approach

* hashset is not necessary, the result is less than 26

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun smallestSubsequence(s: String) = buildString {
        val lastPos = IntArray(26); for (i in s.indices) lastPos[s[i]-'a'] = i
        for (i in s.indices) if (s[i] !in this) {
            while (length > 0 && last() > s[i] && lastPos[last()-'a'] > i)
                setLength(lastIndex)
            append(s[i])
        }
    }
```
```rust
    pub fn smallest_subsequence(s: String) -> String {
        let (mut last, mut res) = ([0; 128], String::new());
        for (i, b) in s.bytes().enumerate() { last[b as usize] = i; }
        for (i, b) in s.bytes().enumerate() { if !res.contains(b as char) {
            while res.bytes().last().map_or(false, |l| l > b && last[l as usize] > i) {
                res.pop();
            }
            res.push(b as char);
        }} res
    }
```

