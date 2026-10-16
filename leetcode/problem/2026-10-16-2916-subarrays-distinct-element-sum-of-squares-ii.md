---
layout: leetcode-entry
title: "2916. Subarrays Distinct Element Sum of Squares II"
permalink: "/leetcode/problem/2026-10-16-2916-subarrays-distinct-element-sum-of-squares-ii/"
leetcode_ui: true
entry_slug: "2026-10-16-2916-subarrays-distinct-element-sum-of-squares-ii"
---

[2916. Subarrays Distinct Element Sum of Squares II](https://leetcode.com/problems/subarrays-distinct-element-sum-of-squares-ii/solutions/8574939/kotlin-by-samoylenkodmitry-sedy/) hard
[substack](https://dmitriisamoilenko.substack.com/p/16102026-2916-subarrays-distinct?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Bozx3GCcXTg)

https://dmitrysamoylenko.com/leetcode/

![16.10.2026.webp](/assets/leetcode_daily_images/16.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1513

#### Problem TLDR

Sum of squares of uniq numbers counts in all subarrays

#### Intuition

Didn't solve.
```j
    // i'm trying to recover from my yesterday's memory
    // x^2 - (x-1)^2 = x^2-(x^2-2x+1) = 2x-1
    //
    // ******j........i
    // ok i'm completely don't know the BIT
```

1. Track last seen index of that number
2. between last seen and the current index we have (i-j) subarrays that gain +1 distinct value
3. math x^2-(x-1)^2=2x-1, this is how square value increases from +1 increase in a single subarray. We have (i-j) of them.
4. BIT tracks prefixes of differences. We updating the range j..i of +1 for the differences. *we have to compensate for prefixes* by maintaining a second tree where we are adding +j (prefix count)

#### Approach

* two trees can be joined into a single with [x,x+1] pairs
* BIT rules: move forward when adding x += x&-x; move back when querying x-=x&-x; always start at i

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
   fun sumCounts(n: IntArray): Int {
        val bit = LongArray((n.size+1)*2); val m = 1000000007
        fun add(i: Int, v: Long) {
            var x = 2*i
            while (x < bit.size) { bit[x] += v; bit[x+1] += v*i; x += x and -x }
        }
        fun query(i: Int): Long {
            var a = 0L; var b = 0L; var x = 2*i
            while (x > 0) { a += bit[x]; b += bit[x+1]; x -= x and -x }
            return (i+1)*a-b
        }
        val seen = IntArray(1000001); var res = 0L; var sum = 0L
        for (i in 1..n.size) {
            val j = seen[n[i-1]]; seen[n[i-1]] = i
            add(j+1, 1); add(i+1, -1)
            val delta = 2*(query(i) - query(j))-(i-j)
            sum = (sum+delta)%m; res = (res + sum)%m
        }
        return (res.toInt()+m)%m
    }
```
```rust

```

