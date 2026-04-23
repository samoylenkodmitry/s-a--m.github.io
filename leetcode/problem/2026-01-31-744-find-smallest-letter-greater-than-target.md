---
layout: leetcode-entry
title: "744. Find Smallest Letter Greater Than Target"
permalink: "/leetcode/problem/2026-01-31-744-find-smallest-letter-greater-than-target/"
leetcode_ui: true
entry_slug: "2026-01-31-744-find-smallest-letter-greater-than-target"
---

[744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/description) easy
[blog post](https://leetcode.com/problems/find-smallest-letter-greater-than-target/solutions/7538983/kotlin-rust-by-samoylenkodmitry-08y8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31012026-744-find-smallest-letter?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KGLwVag6MiU)

![ebdac7eb-6f7b-4fa2-9204-29d66866f239 (1).webp](/assets/leetcode_daily_images/b48a7f10.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1254

#### Problem TLDR

Larger letter in array #easy

#### Intuition

Scan. Or do a binary search.

#### Approach

* or search in t+1..'z'

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 1ms
    fun nextGreatestLetter(l: CharArray, t: Char) =
    l.find{it>t}?:l[0]
    /*
    (l.filter{it>t}+l[0])[0]
    l[l.count{it<=t}%l.size]
    (l.map{it}-('a'..t)+l[0])[0]
    (t+1..'z').find{it in l}?:l[0]
    l[(-1-l.map{it}.binarySearch{if(it>t)1 else -1})%l.size]
    l[(Arrays.binarySearch(l,t+1).let{if(it<0)-1-it else it})%l.size]
    */
```
```rust
// 0ms
    pub fn next_greatest_letter(l: Vec<char>, t: char) -> char {
        l[l.partition_point(|&c|c<=t)%l.len()]
    }
```

