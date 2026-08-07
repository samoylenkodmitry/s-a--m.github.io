---
layout: leetcode-entry
title: "3348. Smallest Divisible Digit Product II"
permalink: "/leetcode/problem/2026-08-07-3348-smallest-divisible-digit-product-ii/"
leetcode_ui: true
entry_slug: "2026-08-07-3348-smallest-divisible-digit-product-ii"
---

[3348. Smallest Divisible Digit Product II](https://leetcode.com/problems/smallest-divisible-digit-product-ii/solutions/8446740/kotlin-rust-by-samoylenkodmitry-18fi/) hard
[substack](https://dmitriisamoilenko.substack.com/p/07082026-3348-smallest-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g6TnbxJgXH0)

https://dmitrysamoylenko.com/leetcode/

![07.08.2026.webp](/assets/leetcode_daily_images/07.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1444

#### Problem TLDR

Smallest n.. digits product divisible by t without zeros

#### Intuition

* Decrease the T for each position in prefix using the T = T/ gcd(T, s[i]).
* go from the end and try to increase the current digit
* use precomputed prefix + current digit + build suffix divisible by T/gcd(T, d), where T is adjusted for this prefix position
* if length of the suffix perfectly matches - this is the result
* if no position adjustement gives good suffix then just build the entire number as a tail with length = N+1

#### Approach

* to build a tail take all digits from 9 to 2 if T%d==0
* zero is stop marker, do not poison prefixes with zero, it would always give T=1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun smallestNumber(n: String, t: Long): String {
        fun g(a: Long, b: Long): Long = if (b==0L)a else g(b, a%b)
        val N=n.length;val p = LongArray(N+1){t}; val z = (n.indexOf('0')+N)%N
        for (i in 0..<if(n[z]=='0')z else N)p[i+1] = p[i]/g(1L*(n[i]-'0'),p[i])
        if (p[N]==1L) return n
        fun tail(t: Long, len: Int): String {
            var t = t; var res = ""
            for (d in 9 downTo 2) while (t%d<1) { t = t/d; res = "$d$res" }
            return if (t > 1) n else res.padStart(len,'1')
        }
        for (i in z downTo 0) for (d in n[i]-'0'+1..9) {
            val e = tail(p[i]/g(p[i],1L*d), N-i-1)
            if (i+1+e.length==N) return n.take(i) + d + e
        }
        return tail(t, N+1).takeIf{it !== n} ?: "-1"
    }
```
```rust

```

