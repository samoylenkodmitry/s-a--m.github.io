---
layout: leetcode-entry
title: "1888. Minimum Number of Flips to Make the Binary String Alternating"
permalink: "/leetcode/problem/2026-03-07-1888-minimum-number-of-flips-to-make-the-binary-string-alternating/"
leetcode_ui: true
entry_slug: "2026-03-07-1888-minimum-number-of-flips-to-make-the-binary-string-alternating"
---

[1888. Minimum Number of Flips to Make the Binary String Alternating](https://open.substack.com/pub/dmitriisamoilenko/p/07032026-1888-minimum-number-of-flips?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/07032026-1888-minimum-number-of-flips?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07032026-1888-minimum-number-of-flips?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/w98s1AmrC9Y)

![48a41980-c2b0-494d-80ae-7e0bba337b75 (1).webp](/assets/leetcode_daily_images/e20c4259.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1290

#### Problem TLDR

Min flips to make 01 alterating string #medium #prefix_sum #sliding_window

#### Intuition

```j
    // 111000
    //  0  1
    //
    // 110001
    // 100011
    //   1
    //      0
    //
    // whats the point of shifting?
    //
    // 110
    // 101 just shifted
    //
    // soo we can start from first 10
    //
    // 011
    // 110
    // 101
    //
    // 11010
    //
    // 10110
    // 01011
    //
    // or double 00
    // 001 shift 010
    //
    // just split 11 idk if it works
    //
    // can be many splits
    // is this dp?
    // 0011
    //
    // 101|101
    //
    // not optimal
    // 10001100101000000|10001100101000000
    //  1010101010101010|1
    //   101010101010101|01
    //    10101010101010|101
    //     1010101010101|0101
    // ...
    //                10|101010101010101
    //                 1|0101010101010101
    // they are all repeating
    //
    // 011|011
    // 010|
    //  01|0
    //   0|10
    // 101|
    //  10|1
    //   1|01 match
    // a  | b
    // miss before + (len-miss) after
```

Prefix sum intuition:
* split at every every index, count bitflips in prefix+suffix(inverted)
* if string length is even return early

Sliding window intuition:
* slide concatenation
* if string length is even slide once
* to move the left pointer flip the bit

#### Approach

* (c+i)%2 gives a match increment

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, or O(n) for prefix sum

#### Code

```kotlin
// 36ms
    fun minFlips(s: String): Int {
        var c = 0
        return (0..<s.length+s.length%2*s.length).minOf { i ->
            c += (s[i%s.length]-'0' + i)%2
            if (i > s.lastIndex) c -= 1-(s[i-s.length]-'0'+i)%2
            if (i < s.lastIndex) s.length else min(c, s.length-c)
        }
    }
```
```rust
// 2ms
    pub fn min_flips(s: String) -> i32 {
        let (n, mut a) = (s.len(), vec![0; s.len()+1]);
        for (i,b) in s.bytes().enumerate() {
            a[i+1]=a[i]+((b+i as u8)&1)as usize }
        if n%2==0 { return a[n].min(n-a[n]) as _ }
        (0..=n).map(|i| {
            let t = a[n] + i - 2 * a[i]; t.min(n - t)
        }).min().unwrap_or(0) as _
    }
```

