---
layout: leetcode-entry
title: "1488. Avoid Flood in The City"
permalink: "/leetcode/problem/2025-10-07-1488-avoid-flood-in-the-city/"
leetcode_ui: true
entry_slug: "2025-10-07-1488-avoid-flood-in-the-city"
---

[1488. Avoid Flood in The City](https://leetcode.com/problems/avoid-flood-in-the-city/description/) medium
[blog post](https://leetcode.com/problems/avoid-flood-in-the-city/solutions/7255787/kotlin-rust-by-samoylenkodmitry-v82f/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07102025-1488-avoid-flood-in-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tNa72g_appw)

![1 (5).webp](/assets/leetcode_daily_images/33f7b554.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1135

#### Problem TLDR

Replace zeros with numbers to avoid duplicates #medium #binary_search #greedy

#### Intuition

```j
    // 0 1 2 3 4 5 6 7 8 9 1011
    // 1 2 0 0 3 4 0 0 3 4 1 2
    //                 i       - first filled, use zero at index 2

    // 0 1 1 - corner case
    //     i -- can't use zero, because no zero before previous 1

    // 0 0 0 0 0 0 2 1 2

    // 0 0 0 0 0 0 2 0 0 1 2 1
    //       ^
    //       can i use any of these? - no

    // 1 2 0 0 3 4 0 0 3 4 1 2
    //             * * i
    //     * *     * *     i      i can use zeros between duplicates
    //                            or, zero can be used for any element before it
    //
    //                            which one to choose? closest

    // 1 2 0 0 2 1
    // 1 2 0 2 0 1
    // 1 0 2 0 2 1

    // 1 2 0 1 0 2
    // 1 0 2 0 1 2

    // 0 1 1
    // i       zi  = [0]
    //   i     fi[1] = 1

    // 0 1 2 3 4 5 6
    // 1 0 2 3 0 1 2
    // i         .     fi[1] = 0
    //   i       .     zi = [1]
    //     i     .     fi[2]=2
    //       i   .     fi[3]=3
    //         i .     zi=[1,4]
    //           i     prev=fi[1]=0, 4>1, res[4]=1, ok so this breaks 2
    //                                              closest is not optimal
    //                                    "smallest after"

```

* remember zero days
* when seeing a duplicate, pick the first zero after the previous duplicate instance

#### Approach

* use the BinarySearch or TreeSet .higher(x)

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 64ms

    fun avoidFlood(r: IntArray): IntArray {
        val zi = TreeSet<Int>(); val fi = HashMap<Int, Int>()
        for ((i, l) in r.withIndex()) if (l > 0) {
            if (fi[l] != null) r[zi.higher(fi[l]) ?: return intArrayOf()] = l
                .also { zi -= zi.higher(fi[l]) }
            fi[l] = i; r[i] = -1
        } else { zi += i; r[i] = 1 }
        return r
    }

```
```rust

// 21ms

    pub fn avoid_flood(mut r: Vec<i32>) -> Vec<i32> {
        let (mut f, mut z) = (HashMap::new(), BTreeSet::new());
        for i in 0..r.len() { let l = r[i];
            if l == 0 { z.insert(i); r[i] = 1; continue }
            if let Some(&j) = f.get(&l) {
                let Some(&d) = z.range(j+1..).next() else { return vec![] };
                r[d] = l; z.remove(&d); }
            f.insert(l, i); r[i] = -1
        } r
    }

```

