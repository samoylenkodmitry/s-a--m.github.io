---
layout: leetcode-entry
title: "3234. Count the Number of Substrings With Dominant Ones"
permalink: "/leetcode/problem/2025-11-15-3234-count-the-number-of-substrings-with-dominant-ones/"
leetcode_ui: true
entry_slug: "2025-11-15-3234-count-the-number-of-substrings-with-dominant-ones"
---

[3234. Count the Number of Substrings With Dominant Ones](https://leetcode.com/problems/count-the-number-of-substrings-with-dominant-ones/description/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-substrings-with-dominant-ones/solutions/7349987/kotlin-rust-by-samoylenkodmitry-8xy9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15112025-3234-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KfcVtvDeWYY)

![3102b2ec-4a3a-4fe0-8739-27fe9096ee7d (2).webp](/assets/leetcode_daily_images/9eb4ca46.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1174

#### Problem TLDR

Subarray zeros^2 less than ones #medium #sliding_window

#### Intuition

```j
    // 000110011(+21 ones tail, can't move j)
    // j      i    z=5 o=3
    //  j     i    z=4 o=3
    //   j    i    z=3 o=3
    //    j   i    z=2 o=3
    // for each i how many 'j' we have
    // z = zeros[i]-zeros[j]
    // o = ones[i]-ones[j]
    // `z*z` less or equal `o`
    // (z[i]-z[j])^2 +o[j] = o[i]
    // z[i]^2 -2z[i]z[j]+z[j]^2+o[j]=o[i]
    // -2z[i]z[j]+z[j]^2+o[j]=o[i]-z[i]^2
    //
    // solve around j, looks like a math problem
    // ok this is a hard problem but let's try brute force O(n*sqrt(n))
    // 32 minute - TLE on input all of ones; because the answer is n^2 of them good
    // hints are not giving any obvious ideas
    // but i suspect we can skip islands of ones
```

#### Approach

* use prefix sums
* use jump array
* if we good we can jump all ones in current island of ones
* if we not, jump to the difference z^2-o

#### Complexity

- Time complexity:
$$O(nsqrt(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 58ms
    fun numberOfSubstrings(s: String): Int {
        val z = IntArray(s.length+1); val o = IntArray(s.length+1)
        val io = IntArray(s.length) { it-1 }; var res = 0
        for (i in 0..<s.length) {
            if (s[i]=='0') ++z[i+1] else ++o[i+1]
            if (i > 0 && s[i]=='1' && s[i-1]=='1') io[i] = io[i-1]
            z[i+1] += z[i]; o[i+1] += o[i]; var j = i
            while (j >= 0) {
                val z = z[i+1]-z[j]; val d = z*z-o[i+1]+o[j]
                if (d > 0) j -= d else { res += j-io[j]; j = io[j] }
            }
        }
        return res
    }
```
```rust
// 63ms
    pub fn number_of_substrings(s: String) -> i32 {
        let (n, s) = (s.len(), s.as_bytes());
        let (mut z, mut o, mut io)=(vec![0;n+1],vec![0;n+1],vec![0;n]);
        let mut r = 0; for i in 0..n { io[i] = i as i32 - 1; }
        for i in 0..n {
            if s[i]==b'0' { z[i+1] = 1 } else { o[i+1] = 1 }
            if i>0 && s[i]==b'1'&&s[i-1]==b'1' { io[i] = io[i-1] }
            z[i+1] += z[i]; o[i+1] += o[i]; let mut j = i;
            while j<n {
                let z = z[i+1]-z[j]; let d = z*z-o[i+1]+o[j];
                if d>0 { j -= d as usize } else { r += j as i32-io[j]; j=io[j]as usize }
            }
        }; r
    }
```

