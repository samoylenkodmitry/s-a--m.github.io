---
layout: leetcode-entry
title: "2654. Minimum Number of Operations to Make All Array Elements Equal to 1"
permalink: "/leetcode/problem/2025-11-12-2654-minimum-number-of-operations-to-make-all-array-elements-equal-to-1/"
leetcode_ui: true
entry_slug: "2025-11-12-2654-minimum-number-of-operations-to-make-all-array-elements-equal-to-1"
---

[2654. Minimum Number of Operations to Make All Array Elements Equal to 1](https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/description) medium
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/solutions/7343417/kotlin-rust-by-samoylenkodmitry-q4w4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12112025-2654-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wTJKGmyNui4)

![f34ff64b-2d89-4f2c-bb07-691e536e4644 (1).webp](/assets/leetcode_daily_images/7a71e2fd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1171

#### Problem TLDR

Min steps to make all '1' by gcd #medium

#### Intuition

* gcd of array = reduce(::gcd)

3 cases:
* have ones in array - then just propagate it in 'size-ones' steps
* gcd(array)>1 - no answer, -1
* gcd(array)==1 - then it is `min_window_gcd1` steps to make a single `1` plus propagate it `size-single ones` steps

#### Approach

* this is a hard problem

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 27ms
    fun minOperations(n: IntArray): Int {
        val n = n.asList(); val ones = n.count { it<2 }
        fun gcd(a: Int, b: Int): Int = if (b==0) a else gcd(b, a%b)
        return if (ones > 0) n.size - ones else if (n.reduce(::gcd) > 1) -1
        else n.size - 2 + (2..n.size).first { L -> n.windowed(L).any { it.reduce(::gcd)<2}}
    }
```
```rust
// 0ms
    pub fn min_operations(n: Vec<i32>) -> i32 {
        let o = n.iter().filter(|&&x|x==1).count() as i32;
        if o > 0 { n.len()as i32 - o} else {
            fn g(mut a:i32,b:&i32)->i32 { let mut b = *b; while b>0 { let t=b;b=a%b;a=t;} a}
            if n.iter().fold(0,g)>1 { -1 } else { n.len() as i32 - 2 +
            (2..=n.len()).find(|&l|n.windows(l).any(|w|w.iter().fold(0,g)==1)).unwrap()as i32}
        }
    }
```

