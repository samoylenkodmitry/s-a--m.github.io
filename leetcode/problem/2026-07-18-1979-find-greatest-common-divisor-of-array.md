---
layout: leetcode-entry
title: "1979. Find Greatest Common Divisor of Array"
permalink: "/leetcode/problem/2026-07-18-1979-find-greatest-common-divisor-of-array/"
leetcode_ui: true
entry_slug: "2026-07-18-1979-find-greatest-common-divisor-of-array"
---

[1979. Find Greatest Common Divisor of Array](https://leetcode.com/problems/find-greatest-common-divisor-of-array/solutions/8404572/kotlin-rust-by-samoylenkodmitry-nj6r/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18072026-1979-find-greatest-common?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ovJUXzzYnvw)

https://dmitrysamoylenko.com/leetcode/

![18.07.2026.webp](/assets/leetcode_daily_images/18.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1424

#### Problem TLDR

GCD of min & max

#### Intuition

Iterate and find the min, max, then the gcd.

#### Approach

* remember gcd: ab bab means a/b b should not be 0, then the recursive call of b,a%b
* Rust: sort gives the shortest code; also there is a minmax from itertools
* Kotlin: make the entire solution recursive

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun findGCD(n: IntArray, a:Int=n.max(), b:Int=n.min()):Int=
        if(b==0)a else findGCD(n, b,a%b)
```
```rust
    pub fn find_gcd(n: Vec<i32>) -> i32 {
        let (mut b, mut a) = n.into_iter().minmax().into_option().unwrap();
        while b > 0 { (a, b) = (b, a % b) } a
    }
```

