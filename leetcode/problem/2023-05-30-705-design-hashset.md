---
layout: leetcode-entry
title: "705. Design HashSet"
permalink: "/leetcode/problem/2023-05-30-705-design-hashset/"
leetcode_ui: true
entry_slug: "2023-05-30-705-design-hashset"
---

[705. Design HashSet](https://leetcode.com/problems/design-hashset/description/) easy
[blog post](https://leetcode.com/problems/design-hashset/solutions/3577326/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/28052023-705-design-hashset?sd=pf)
#### Telegram
https://t.me/leetcode_daily_unstoppable/228
#### Problem TLDR
Write a `HashSet`.
#### Intuition
There are different [hash functions](https://en.wikipedia.org/wiki/Hash_function). Interesting implementations is In Java `HashMap` [https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/HashMap.java](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/HashMap.java)

#### Approach
Use `key % size` for the hash function, grow and rehash when needed.

#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

class MyHashSet(val initialSz: Int = 16, val loadFactor: Double = 1.6) {
            var buckets = Array<LinkedList<Int>?>(initialSz) { null }
            var size = 0

            fun hash(key: Int): Int = key % buckets.size

            fun rehash() {
            if (size > buckets.size * loadFactor) {
                val oldBuckets = buckets
                buckets = Array<LinkedList<Int>?>(buckets.size * 2) { null }
                    oldBuckets.forEach { it?.forEach { add(it) } }
                }
            }

            fun bucket(key: Int): LinkedList<Int> {
                val hash = hash(key)
                if (buckets[hash] == null) buckets[hash] = LinkedList<Int>()
                    return buckets[hash]!!
                }

                fun add(key: Int) {
                    val list = bucket(key)
                    if (!list.contains(key)) {
                        list.add(key)
                        size++
                        rehash()
                    }
                }

                fun remove(key: Int) {
                    bucket(key).remove(key)
                }

                fun contains(key: Int): Boolean =
                   bucket(key).contains(key)
}

```

